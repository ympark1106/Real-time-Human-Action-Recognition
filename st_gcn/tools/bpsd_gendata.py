import numpy as np
import argparse
import os
import sys
import json
import pickle
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from st_gcn.feeder.feeder_bpsd import Feeder_bpsd
from numpy.lib.format import open_memmap

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

def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        filtered_label_path,
        num_person_in=5,  # observe the first 5 persons
        num_person_out=2,  # then choose 2 persons with the highest score
        max_frame=300):

    # Load filtered labels
    with open(filtered_label_path, 'r') as f:
        filtered_labels = json.load(f)

    # Get a list of JSON files in the data directory
    sample_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

    sample_name = []
    sample_label = []

    # Create output file for data
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(filtered_labels), 3, max_frame, 18, num_person_out))

    # Process each sample
    for i, (key, value) in enumerate(filtered_labels.items()):
        # Check if skeleton exists and is valid
        if not value["has_skeleton"]:
            print(f"Skipping {key}, skeleton not available in JSON.")
            continue
        
        # Check if the file exists in the data directory
        json_file = f"{key}.json"
        if json_file not in sample_files:
            print(f"Skipping {key}, skeleton data not found in data directory.")
            continue

        # Load skeleton data from JSON
        with open(os.path.join(data_path, json_file), 'r') as f:
            video_info = json.load(f)

        # Prepare data array
        data_numpy = np.zeros((3, max_frame, 18, num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info['skeleton']):
                if m >= num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # Centralize data
        data_numpy[0:2] -= 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # Save processed data
        fp[len(sample_name), :, :, :, :] = data_numpy[:, :max_frame, :, :num_person_out]
        sample_name.append(json_file)
        sample_label.append(value["label"])

        # Print progress
        print_toolbar(i * 1.0 / len(filtered_labels),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(filtered_labels)))

    # Save labels
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    print(f"Finished processing. Total samples saved: {len(sample_name)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter for BPSD.')
    parser.add_argument(
        '--data_path', default='data/Kinetics/kinetics-skeleton/')    
    parser.add_argument(
        '--out_folder', default='data/Kinetics/kinetics-skeleton/bpsd')
    parser.add_argument(
        '--filtered_label_path', default='data/Kinetics/kinetics-skeleton/kinetics_train_label_bpsd.json',
        help='Path to the filtered JSON file with BPSD labels.')
    arg = parser.parse_args()

    part = [
        'train', 
        # 'val'
            ]
    for p in part:
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path, arg.filtered_label_path)
