import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  # x_lens = [len(x) for x in xx]
  # y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy_pad

class PyTorchDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for pytorch

    This does just two things:
        1. On-demand normalization
        2. Returns torch.tensor instead of ndarray
    """

    word_to_num = None
    target_to_num = None
    num_to_target = None

    def __init__(self, path):
        X, y = PyTorchDataset.load_file(path)

        if self.word_to_num is None:
            vocab = np.unique(np.append(X, ['UUUNNNKKK']))
            unique_target = np.unique(y)
            PyTorchDataset.word_to_num = dict(zip(vocab, range(len(vocab))))
            PyTorchDataset.target_to_num = dict(zip(unique_target, range(len(unique_target))))
            PyTorchDataset.num_to_target = {k:v for v,k in self.target_to_num.items()}
        self.sequences, self.targets = PyTorchDataset.create_sentences(X, y,
                                                                       PyTorchDataset.word_to_num,
                                                                       PyTorchDataset.target_to_num)
        # print(self.sequences[:2])
        # print(self.targets[:2])

    def __getitem__(self, idx):
        x, y = self.sequences[idx], self.targets[idx]

        data = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(y)
        return (data, target)

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def load_file(path):
        data, target = np.genfromtxt(path, dtype='U20', unpack=True)
        return data, target

    @staticmethod
    def create_sentences(data, target, word_to_num, target_to_num):
        ind = list(np.where(data == '.')[0] + 1)
        arr_per_sequences = np.split(data, ind)
        target_per_sequences = np.split(target, ind)

        sentences = [[word_to_num.get(word, word_to_num['UUUNNNKKK']) for word in seq] for seq in arr_per_sequences]
        targets = [[target_to_num.get(word) for word in seq] for seq in target_per_sequences]
        return sentences, targets


if __name__ == '__main__':
    # Must change
    #root = Path('DL_3/Part3/{}'.format(sys.argv[1]))
    train_file = sys.argv[1]

    dataset = PyTorchDataset(train_file)

    dataloader_train = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=pad_collate)

    for i, batch in enumerate(dataloader_train):
        data, labels = batch
        print(data)
        #print(labels.shape)
