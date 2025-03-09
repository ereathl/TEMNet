import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SampleReader:

    def __init__(self, file_name):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        project_root = os.path.join(current_dir, os.pardir)
        datasets_path = os.path.join(project_root, 'Datasets')
        self.seq_path = datasets_path + '/' + file_name + '/Sequence/'
        self.shape_path = datasets_path + '/' + file_name + '/shape/'

    def get_seq(self, k, Test=False):

        if Test is False:
            row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=' ', header=None)
        else:
            row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=' ', header=None)
        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])
        new_seq_len = seq_len - k + 1
        completed_seqs = np.zeros(shape=(seq_num, new_seq_len, 4 * k))
        completed_labels = np.zeros(shape=(seq_num, 1))
        for i in range(seq_num):
            completed_seqs[i] = one_hot(row_seq.loc[i, 1], k)
            completed_labels[i] = row_seq.loc[i, 2]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])
        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'train' + '_' + shape + '.csv'))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'test' + '_' + shape + '.csv'))

        completed_shape = np.zeros(shape=(len(shapes), shape_series[0].shape[0], shape_series[0].shape[1]))

        for i in range(len(shapes)):
            completed_shape[i] = shape_series[i]
        completed_shape = np.transpose(completed_shape, [1, 0, 2])

        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape

class Datasets(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ["EP", "HelT", "MGW", "ProT", "Roll"]
        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(3, Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)

    def __getitem__(self, idx):
        return self.completed_seqs[idx], self.completed_shape[idx], self.completed_labels[idx]

    def __len__(self):
        return self.completed_seqs.shape[0]

def one_hot(seq, k):
    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    seq_len = len(seq)
    num_kmers = seq_len - k + 1

    code = np.empty(shape=(num_kmers, 4 * k))

    for i in range(num_kmers):
        kmer = seq[i:i+k]
        code[i] = np.concatenate([base_map[base] for base in kmer])

    return code