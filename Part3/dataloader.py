import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

# class CustomerDataSet(Dataset):
#     def __init__(self, path, encoded_words=None, encoded_labels=None):
#         pre = Preprocessing(path)
#
#         if not encoded_words:
#             vocab, labels, encoded_words, encoded_labels, encode_num_to_labels = pre.get_vocab_and_labels()
#             tokens = pre.load_data()
#             contexts = pre.get_contexts(tokens, encoded_words, encoded_labels)
#
#             self.sample, self.label = list(zip(*contexts))
#             self.vocab = vocab
#             self.labels = labels
#             self.encoded_words = encoded_words
#             self.encoded_labels= encoded_labels
#             self.encode_num_to_label = encode_num_to_labels
#         else:
#             encoded_words = encoded_words
#             encoded_labels = encoded_labels
#             tokens = pre.load_data()
#             contexts = pre.get_contexts(tokens, encoded_words, encoded_labels)
#             self.sample, self.label = list(zip(*contexts))
#
#     def __getitem__(self, index):
#
#         label = torch.tensor(self.label[index], dtype=torch.long)
#         sample = torch.tensor(self.sample[index])
#         return sample, label
#
#     def __len__(self):
#         return len(self.sample)

import numpy as np
class PyTorchDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for pytorch

    This does just two things:
        1. On-demand normalization
        2. Returns torch.tensor instead of ndarray
    """
    def __init__(self, path):
        X, y = PyTorchDataset.load_file(path)
        sequences, targets = PyTorchDataset.create_sentences(X, y)

        vocab = np.unique(X)
        unique_target = np.unique(y)



    def __getitem__(self, idx):
        # x, y = self.X[idx], self.Y[idx]
        # x = minmax_scale(x, self.X_min, self.X_max, feature_range=(0.01, 0.99))
        # y = scale(y, self.Y_mean, self.Y_scale)
        # l = torch.from_numpy(self.lengths[idx])
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, l

    def __len__(self):
        return len(self.X)

    @staticmethod
    def load_file(path):
        data, target = np.genfromtxt(path, dtype='U20', unpack=True)

        return data, target

    @staticmethod
    def create_sentences(data, target):
        ind = list(np.where(data == '.')[0] + 1)
        arr_per_sequences = np.split(data, ind)
        sentences = [' '.join(seq) for seq in arr_per_sequences]

        target_per_sequences = np.split(target, ind)
        targets = [' '.join(seq) for seq in target_per_sequences]
        print(sentences[:2])
        print(targets[:2])
        return sentences, targets

if __name__ == '__main__':
    #Must change
    root = Path('DL_3/Part3/{}'.format(sys.argv[1]))

    # Path to data files
    train_file = root / "train"

    data = PyTorchDataset('/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/pos/train')