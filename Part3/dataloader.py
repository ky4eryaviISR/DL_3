
import torch
from torch import cuda
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from torch.utils.data.dataloader import DataLoader


device = 'cuda' if cuda.is_available() else 'cpu'

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).to(device)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0).to(device)

  return xx_pad, yy_pad, x_lens, None


def padding(x, max):
    diff = max - len(x)
    return x + [0] * diff


def pad_part_2(batch):
    X, X2, Y, sentence_length, length_word = zip(*batch)
    max_word = max([j for i in length_word for j in i])
    max_sentence = max(sentence_length)
    word = torch.zeros((len(batch), max_sentence, max_word), dtype=torch.long).to(device=device)
    sentence = torch.zeros((len(batch), max_sentence), dtype=torch.long).to(device=device)
    yy_pad = torch.zeros((len(batch), max_sentence), dtype=torch.long).to(device=device)
    length_word = sum([padding(i, max_sentence) for i in length_word], [])
    for i, item in enumerate(zip(X, X2, Y)):
        sen, w, y = item
        w = [padding(i, max_word) for i in w]
        word[i, :sentence_length[i], :max_word] = torch.Tensor(w)
        sentence[i, :sentence_length[i]] = torch.Tensor(sen)
        yy_pad[i, :sentence_length[i]] = torch.Tensor(y)

    return (sentence, word), yy_pad, torch.tensor(sentence_length), torch.tensor(length_word)


def pad_part_3(batch):
    X, Y, sentence_length, length_word = zip(*batch)
    max_word = max([j for i in length_word for j in i])
    max_sentence = max(sentence_length)
    word = torch.zeros((len(batch), max_sentence, max_word), dtype=torch.long).to(device=device)
    yy_pad = torch.zeros((len(batch), max_sentence), dtype=torch.long).to(device=device)
    length_word = sum([padding(i, max_sentence) for i in length_word], [])
    for i, item in enumerate(zip(X, Y)):
        w, y = item
        w = [padding(i, max_word) for i in w]
        word[i, :sentence_length[i], :max_word] = torch.Tensor(w)
        yy_pad[i, :sentence_length[i]] = torch.Tensor(y)

    return word, yy_pad, torch.tensor(sentence_length), torch.tensor(length_word)

def pad_collate_sorted(batch):
    if PyTorchDataset.variation == 'd':
        return pad_part_2(batch)
    if PyTorchDataset.variation == 'b':
        return pad_part_3(batch)
    (xx, yy) = zip(*batch)
    if xx[0].dim() == 1 or PyTorchDataset.variation == 'c':
        return pad_collate(batch)

    x_lens = [len(x) for x in xx]
    shape_0 = max([item.shape[0] for item in xx])
    shape_1 = max([item.shape[1] for item in xx])
    xx_pad = torch.zeros(len(batch), shape_0, shape_1, dtype=torch.long).to(device)
    yy_pad = torch.zeros(len(batch), shape_0, dtype=torch.long).to(device)
    for i, item in enumerate(zip(xx, yy)):
        x, y = item
        xx_pad[i, :x.shape[0], :x.shape[1]] = x
        yy_pad[i, :x.shape[0]] = y
    word_lens = []
    for k in range(xx_pad.shape[0]):
        for i in range(xx_pad.shape[1]):
            for j in range(xx_pad.shape[2]):
                if xx_pad[k][i][j] == 0:
                    word_lens.append(j)
                    break
            else:
                word_lens.append(j+1)
    return xx_pad, yy_pad, torch.tensor(x_lens).to(device), torch.tensor(word_lens).to(device)


class PyTorchDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for pytorch

    This does just two things:
        1. On-demand normalization
        2. Returns torch.tensor instead of ndarray
    """
    variation = None
    word_to_num = None
    target_to_num = None
    num_to_target = None
    vocab = None

    def __init__(self, path, variation='a'):
        sentences, targets = self.load_data(path)
        train = False
        if not PyTorchDataset.word_to_num:
            self.create_dictionaries(sentences, targets)
            PyTorchDataset.variation = variation
            train = True

        self.X, self.Y = self.convert2num(sentences, targets, train)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        data = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(y, dtype=torch.long)
        return (data, target)

    def convert2num(self, sentences, targets, train):

        num_sentences = []
        for sen in sentences:
            sen_temp = []
            words = sen.split()
            for www in words:
                if train and PyTorchDataset.vocab[www] < 5:
                    www = 'UNK'
                elif not train and www not in PyTorchDataset.word_to_num.keys():
                    if www.lower() in PyTorchDataset.word_to_num.keys():
                        www = www.lower()
                    else:
                        www = 'UNK'
                sen_temp.append(PyTorchDataset.word_to_num.get(www, PyTorchDataset.word_to_num['UNK']))
            num_sentences.append(sen_temp)

        num_targets = []
        for tar in targets:
            tar_temp = []
            target = tar.split()
            for ttt in target:
                tar_temp.append(PyTorchDataset.target_to_num.get(ttt))
            num_targets.append(tar_temp)

        return num_sentences, num_targets

    def create_dictionaries(cls, sentences, targets):
        # toDo Counter and filtering vocab

        sen_temp = ' '.join(sentences).split()
        tar_temp = ' '.join(targets).split()

        PyTorchDataset.vocab = Counter(sen_temp)
        PyTorchDataset.vocab['UNK'] = 999
        targets = set(tar_temp)

        PyTorchDataset.word_to_num = dict(zip(PyTorchDataset.vocab.keys(), range(1, len(PyTorchDataset.vocab)+1)))
        PyTorchDataset.word_to_num['<PAD>'] = 0
        PyTorchDataset.target_to_num = dict(zip(targets, range(1, len(targets)+1)))
        PyTorchDataset.target_to_num['<PAD>'] = 0
        PyTorchDataset.num_to_target = {k: v for v, k in cls.target_to_num.items()}

    def load_data(self, path):
        with open(path) as file:
            temp_sentences = []
            temp_targets = []
            sentences = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences.append(word), temp_targets.append(label)
                except ValueError:
                    sequens_sen = ' '.join(temp_sentences)
                    sequens_tar = ' '.join(temp_targets)

                    sentences.append(sequens_sen)
                    targets.append(sequens_tar)

                    temp_sentences = []
                    temp_targets = []

        return sentences, targets


class CharDataset(PyTorchDataset):

    def __init__(self, path, variation=None):
        self.sentences_var = []
        self.length_word = []
        self.sentence_length = []
        super(CharDataset, self).__init__(path, variation)

    def load_data(self, path):
        with open(path) as file:
            temp_sentences_char = []
            sentences_char = []
            temp_targets = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences_char.append(list(word))
                    temp_targets.append(label)
                except ValueError:
                    self.length_word.append([len(row) for row in temp_sentences_char])
                    self.sentence_length.append(len(temp_sentences_char))
                    self.sentences_var.append(temp_sentences_char)
                    sequens_sen_char = ' '.join(set([c for row in temp_sentences_char for c in row]))
                    sequens_tar = ' '.join(temp_targets)
                    sentences_char.append(sequens_sen_char)
                    targets.append(sequens_tar)

                    temp_targets = []
                    temp_sentences_char = []

        return sentences_char, targets

    def convert2num(self, sentences, targets, train):
        num_sentences = []
        for sen in self.sentences_var:
            sen_temp = []
            for char_lst in sen:
                sen_temp.append([PyTorchDataset.word_to_num[ch] for ch in char_lst])
            num_sentences.append(sen_temp)

        num_targets = []
        for tar in targets:
            tar_temp = []
            target = tar.split()
            for ttt in target:
                tar_temp.append(PyTorchDataset.target_to_num.get(ttt))
            num_targets.append(tar_temp)

        return num_sentences, num_targets

    def padding(self,x,max):
        diff = max - len(x)
        return x + [0]*diff

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sentence_length[idx], self.length_word[idx]


class PyTorchDataset_C(PyTorchDataset):

    def __init__(self, path, variation):
        self.sentences_var = []
        super(PyTorchDataset_C, self).__init__(path, variation)

    def load_data(self, path):
        with open(path) as file:
            temp_sentences = []
            temp_targets = []
            sentences = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences.append([word, word[:3], word[-3:]]), temp_targets.append(label)
                except ValueError:
                    self.sentences_var.append(temp_sentences)
                    sequens_sen = ' '.join(set([word for row in temp_sentences for word in row ]))
                    sequens_tar = ' '.join(temp_targets)
                    sentences.append(sequens_sen)
                    targets.append(sequens_tar)

                    temp_sentences = []
                    temp_targets = []

        return sentences, targets

    def convert2num(self, sentences, targets, train):
        num_sentences = []
        for sen in self.sentences_var:
            sen_temp = []
            for word, pref, suff in sen:
                if train and PyTorchDataset.vocab[word] < 5:
                    word = 'UNK'
                elif not train and word not in PyTorchDataset.word_to_num.keys():
                    if word.lower() in PyTorchDataset.word_to_num.keys():
                        word = word.lower()
                    else:
                        word = 'UNK'
                sen_temp.append([PyTorchDataset.word_to_num[word],
                                PyTorchDataset.word_to_num.get(pref, PyTorchDataset.word_to_num['UNK']),
                                PyTorchDataset.word_to_num.get(suff, PyTorchDataset.word_to_num['UNK'])])
            num_sentences.append(sen_temp)

        num_targets = []
        for tar in targets:
            tar_temp = []
            target = tar.split()
            for ttt in target:
                tar_temp.append(PyTorchDataset.target_to_num.get(ttt))
            num_targets.append(tar_temp)

        return num_sentences, num_targets


class CharSentenceDataset(PyTorchDataset):
    word_to_num = {}
    vocab = {}

    def __init__(self, path, variation='a'):
        self.sentences_var = []
        self.word_sentences = []
        self.sentence_length = []
        self.length_word = []
        sentences, targets = self.load_data(path)
        train = False
        if not PyTorchDataset.word_to_num:
            self.create_dictionaries(sentences, targets)
            PyTorchDataset.variation = variation
            train = True

        self.X, self.X2, self.Y = self.convert2num(sentences, targets, train)

    def load_data(self, path):
        with open(path) as file:
            temp_sentences = []
            temp_targets = []
            temp_sentences_char = []
            sentences = []
            sentences_char = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences_char.append(list(word))
                    temp_sentences.append(word)
                    temp_targets.append(label)
                except ValueError:
                    self.length_word.append([len(row) for row in temp_sentences_char])
                    self.sentence_length.append(len(temp_sentences))
                    self.sentences_var.append(temp_sentences_char)
                    self.word_sentences.append(temp_sentences)
                    sequens_sen_char = ' '.join(set([c for row in temp_sentences_char for c in row]))
                    sequens_sen = ' '.join(temp_sentences)
                    sequens_tar = ' '.join(temp_targets)
                    sentences.append(sequens_sen)
                    sentences_char.append(sequens_sen_char)
                    targets.append(sequens_tar)

                    temp_sentences = []
                    temp_targets = []
                    temp_sentences_char = []

        return (sentences, sentences_char), targets

    def create_dictionaries(self, sentences, targets):
        sen_temp = ' '.join(sentences[0]).split()
        sen_temp_char = ' '.join(sentences[1]).split()
        tar_temp = ' '.join(targets).split()

        PyTorchDataset.vocab = Counter(sen_temp)
        PyTorchDataset.vocab['UNK'] = 999
        targets = set(tar_temp)

        PyTorchDataset.word_to_num = dict(
            zip(PyTorchDataset.vocab.keys(), range(1, len(PyTorchDataset.vocab) + 1)))
        PyTorchDataset.word_to_num['<PAD>'] = 0
        PyTorchDataset.target_to_num = dict(zip(targets, range(1, len(targets) + 1)))
        PyTorchDataset.target_to_num['<PAD>'] = 0
        PyTorchDataset.num_to_target = {k: v for v, k in self.target_to_num.items()}

        CharSentenceDataset.vocab = Counter(sen_temp_char)
        CharSentenceDataset.vocab['UNK'] = 999

        CharSentenceDataset.word_to_num = dict(
            zip(CharSentenceDataset.vocab.keys(), range(1, len(CharSentenceDataset.vocab) + 1)))
        CharSentenceDataset.word_to_num['<PAD>'] = 0

    def convert2num(self, sentences, targets, train):
        num_sentences_char = []
        for sen in self.sentences_var:
            sen_temp = []
            for char_lst in sen:
                sen_temp.append([CharSentenceDataset.word_to_num[ch] for ch in char_lst])
            num_sentences_char.append(sen_temp)

        num_sentences_word = []
        for sen in self.word_sentences:
            sen_temp = []
            for www in sen:
                if train and PyTorchDataset.vocab[www] < 5:
                    www = 'UNK'
                elif not train and www not in PyTorchDataset.word_to_num.keys():
                    if www.lower() in PyTorchDataset.word_to_num.keys():
                        www = www.lower()
                    else:
                        www = 'UNK'
                sen_temp.append(PyTorchDataset.word_to_num.get(www, PyTorchDataset.word_to_num['UNK']))
            num_sentences_word.append(sen_temp)

        num_targets = []
        for tar in targets:
            tar_temp = []
            target = tar.split()
            for ttt in target:
                tar_temp.append(PyTorchDataset.target_to_num.get(ttt))
            num_targets.append(tar_temp)

        return num_sentences_word, num_sentences_char, num_targets

    def __getitem__(self, idx):
        return self.X[idx], self.X2[idx], self.Y[idx], self.sentence_length[idx], self.length_word[idx]
