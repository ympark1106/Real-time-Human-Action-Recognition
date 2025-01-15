# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder_bpsd(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset with BPSD labels
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
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
        # Load file list from data directory
        self.sample_name = os.listdir(self.data_path)
        if self.debug:
            self.sample_name = self.sample_name[:2]

        # Load label info (JSON format with specific structure)
        with open(self.label_path, 'r') as f:
            label_info = json.load(f)

        print(f"Total samples in data directory: {len(self.sample_name)}")
        print(f"Total labels in label file: {len(label_info)}")

        # Filter only files present in both JSON and data path
        sample_ids_data = set([name.split('.')[0] for name in self.sample_name])
        sample_ids_label = set(label_info.keys())
        matching_ids = sample_ids_data & sample_ids_label  # Intersection of IDs

        # Update sample_name and labels
        self.sample_name = [name for name in self.sample_name if name.split('.')[0] in matching_ids]
        self.label = np.array([label_info[name.split('.')[0]]["label_index"] for name in self.sample_name])

        print(f"Valid samples after filtering: {len(self.sample_name)}")
        print(f"Matching IDs: {len(matching_ids)}")

        # Data dimensions
        self.N = len(self.sample_name)  # Number of samples
        self.C = 3  # Channels
        self.T = 300  # Frames
        self.V = 18  # Joints
        self.M = self.num_person_out  # Max number of persons

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['bpsd_label']  # Use 'bpsd_label' instead of 'label_index'
        assert (self.label[index] == label)

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # match poses between 2 frames
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def test(data_path, label_path, vid=None, graph=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder_(
            data_path, label_path, pose_matching=False, num_person=10),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is None:
        index = 0
    elif type(vid) == int:
        index = vid
    else:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)

    name = loader.dataset.sample_name[index]

    data, label = loader.dataset[index]
    data = data.reshape(data.shape)

    # for batch_idx, (data, label) in enumerate(loader):
    C, T, V, M = data.shape
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if graph is None:
        p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
        pose = [
            ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
        ]
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            # print t
            for m in range(M):
                pose[m].set_xdata(data[0, t, :, m])
                pose[m].set_ydata(data[1, t, :, m])
            fig.canvas.draw()
            plt.pause(0.001)
            # raw_input(t)
    else:
        p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
        import sys
        from os import path
        sys.path.append(
            path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        G = import_class(graph)()
        edge = G.inward
        pose = []
        for m in range(M):
            a = []
            for i in range(len(edge)):
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
            pose.append(a)
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            for m in range(M):
                for i, (v1, v2) in enumerate(edge):
                    pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                    pose[m][i].set_ydata(-data[1, t, [v1, v2], m])
            fig.canvas.draw()
            plt.pause(0.001)
            # raw_input(t)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    data_path = './data/Kinetics/kinetics-skeleton/kinetics_val'
    label_path = './data/Kinetics/kinetics-skeleton/kinetics_val_label_bpsd.json'
    graph = 'st_gcn.graph.Kinetics'
    # test(data_path, label_path, vid='iqkx0rrCUCo', graph=graph)
    test(data_path, label_path, vid=11111, graph=graph)
    # test(data_path, label_path, vid = 11199, graph=graph)
