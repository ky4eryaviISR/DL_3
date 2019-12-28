import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import sys
import numpy as np
from collections import Counter


def pad_collate(batch):
  (xx, yy) = zip(*batch)

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

    def __getitem__(self, idx):
        x, y = self.sequences[idx], self.targets[idx]

        data = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(y, dtype=torch.long)
        return (data, target)

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def load_file(path):
        data, target = np.genfromtxt(path, dtype='U20', unpack=True)

        word_dict = Counter(data)
        min_threshold = 3

        newDict = dict(filter(lambda elem: elem[1] > min_threshold, word_dict.items()))
        keys = [(word if word in newDict else 'UUUNNNKKK', target) for word, target in zip(data, target)]
        data, target = list(zip(*(keys)))
        return np.array(data), np.array(target)

    @staticmethod
    def create_sentences(data, target, word_to_num, target_to_num):
        ind = list(np.where(data == '.')[0] + 1)
        arr_per_sequences = np.split(data, ind)[:-1]
        target_per_sequences = np.split(target, ind)

        sentences = [[word_to_num.get(word, word_to_num['UUUNNNKKK']) for word in seq] for seq in arr_per_sequences]
        targets = [[target_to_num.get(word) for word in seq] for seq in target_per_sequences]
        return sentences, targets


class CharDataset(PyTorchDataset):

    def __init__(self, path):
        super().__init__(path)

        X, y = PyTorchDataset.load_file(path)
        char = CharDataset.preprocessing_char(X)
        target = [self.target_to_num.get(item) for item in y]
        print(target[:2])
        print(char[:2])


    @staticmethod
    def preprocessing_char(X):

        char_vocab = set([char for word in X for char in word])
        char_vocab.add('UNK')
        char_to_num = dict(zip(char_vocab, range(len(char_vocab))))

        char_list = [[char_to_num.get(char, 'UNK') for char in word] for word in X]

        return char_list



if __name__ == '__main__':
    # Must change
    #root = Path('DL_3/Part3/{}'.format(sys.argv[1]))
    train_file = sys.argv[1]
    # train_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train'
    dataset = PyTorchDataset(train_file)

    dataloader_train = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=pad_collate)

    for i, batch in enumerate(dataloader_train):
        data, labels = batch
        # print(data)
        print(data.shape)
