import numpy as np
import argparse
import os
import sys
import torch
from ntu_read_skeleton import read_xyz
from numpy.lib.format import open_memmap
import pickle

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview_select',
            part='eval',
            selected_classes=None,
            device='cpu'):
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if not filename.endswith('.skeleton'):
            print(f"Skipping non-skeleton file: {filename}")
            continue

        if 'A' not in filename or 'P' not in filename or 'C' not in filename:
            print(f"Invalid file format: {filename}")
            continue

        try:
            action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
        except ValueError as e:
            print(f"Error parsing file: {filename}, error: {e}")
            continue

        if selected_classes is not None and action_class not in selected_classes:
            continue

        if benchmark == 'xview_select':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub_select':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not istraining
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        data = torch.tensor(data, device=device)  # Move data to specified device
        fp[i, :, 0:data.shape[1], :, :] = data.cpu().numpy()  # Move back to CPU for saving
    end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NTU-RGB-D/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='data/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU-RGB-D')
    parser.add_argument('--selected_classes', nargs='+', type=int, default=None,
                        help='List of action classes to include (e.g., 43 50 for falling down and punch/slap).')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        help='Device to use for processing (cpu or cuda).')
    parser.add_argument('-g','--gpu_id', type=int, default=0, 
                        help='GPU ID to use if device is set to cuda.')

    benchmark = ['xsub_select', 'xview_select']
    part = ['train', 'val']
    arg = parser.parse_args()

    # Set the device
    if arg.device == 'cuda':
        device = torch.device(f"cuda:{arg.gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                selected_classes=arg.selected_classes,
                device=device)
