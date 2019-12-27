import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

        vocab = np.unique(X)
        unique_target = np.unique(y)

        word_to_num = dict(zip(vocab, range(len(vocab))))
        target_to_num = dict(zip(unique_target, range(len(unique_target))))

        self.sequences, self.targets = PyTorchDataset.create_sentences(X, y, word_to_num, target_to_num)
        print(self.sequences[:2])
        print(self.targets[:2])


    def __getitem__(self, idx):
        x, y = self.sequences[idx], self.targets[idx]


        data = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(y)
        return data, target

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

        sentences = []
        for seq in arr_per_sequences:
            temp = []
            for word in seq:
                temp.append(word_to_num.get(word))
            sentences.append(temp)

        targets = []
        for seq in target_per_sequences:
            temp = []
            for word in seq:
                temp.append(target_to_num.get(word))
            targets.append(temp)

        # sentences = [' '.join(word_to_num.get(word)) for seq in arr_per_sequences for word in seq]
        #
        # target_per_sequences = np.split(target, ind)
        # targets = [' '.join(target_to_num.get(tar)) for seq in target_per_sequences for tar in target]
        return sentences, targets

if __name__ == '__main__':
    # Must change
    root = Path('DL_3/Part3/{}'.format(sys.argv[1]))

    # Path to data files
    train_file = root / "train"

    dataset = PyTorchDataset('/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train')

    dataloader_train = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(dataloader_train):
        data, labels = batch
        print(data, labels)
