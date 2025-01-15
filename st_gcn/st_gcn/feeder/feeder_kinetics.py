# sys
import os
import sys
import numpy as np
import pickle
import json
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable

# visualization
import time

# operation
from . import tools

class Feeder_kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label in '.pkl' format
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the beginning or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If true, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=2,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # Load data from .npy file
        print("Loading data from .npy file...")
        self.data = np.load(self.data_path, mmap_mode='r')  # Memory efficient loading

        # Load labels from .pkl file
        print("Loading labels from .pkl file...")
        with open(self.label_path, 'rb') as f:
            label_info = pickle.load(f)
        self.sample_name, self.label = label_info  # Unpack names and labels

        # Ensure skeleton availability
        if self.ignore_empty_sample:
            valid_indices = [i for i, name in enumerate(self.sample_name) if self.label[i] is not None]
            self.sample_name = [self.sample_name[i] for i in valid_indices]
            self.label = np.array([self.label[i] for i in valid_indices])
            self.data = self.data[valid_indices]

        # Update output dimensions
        self.N, self.C, self.T, self.V, self.M = self.data.shape

        print(f"Data loaded: {self.N} samples, {self.C} channels, {self.T} frames, {self.V} joints, {self.M} persons")

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, index):
        # Retrieve the data and label for the given index
        data_numpy = self.data[index]
        label = self.label[index]
        
        if isinstance(label, tuple):
            label = label[0]  # 튜플에서 첫 번째 값을 가져옴

        # Data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # Match poses between two frames if enabled
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


if __name__ == '__main__':
    data_path = './data/Kinetics/kinetics-skeleton/bpsd/train_data.npy'
    label_path = './data/Kinetics/kinetics-skeleton/bpsd/train_label.pkl'

    dataset = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        random_choose=True,
        random_shift=True,
        window_size=150
    )

    print(f"Number of samples: {len(dataset)}")

    for i in range(3):  # Show first 3 samples
        data, label = dataset[i]
        print(f"Sample {i}: data shape = {data.shape}, label = {label}")
