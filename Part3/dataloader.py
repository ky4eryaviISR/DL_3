
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy_pad, x_lens, y_lens


class PyTorchDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for pytorch

    This does just two things:
        1. On-demand normalization
        2. Returns torch.tensor instead of ndarray
    """
    word_to_num = None
    target_to_num = None
    num_to_target = None
    vocab = None

    def __init__(self, path):
        sentences, targets = self.load_data(path)
        if not PyTorchDataset.word_to_num:
            self.create_dictionaries(sentences, targets)

        self.X, self.Y = self.convert2num(sentences, targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        data = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(y, dtype=torch.long)
        return (data, target)

    def convert2num(self, sentences, targets):

        num_sentences = []
        for sen in sentences:
            sen_temp = []
            words = sen.split()
            for www in words:
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

        PyTorchDataset.vocab = set(sen_temp)
        PyTorchDataset.vocab.add('UNK')
        PyTorchDataset.vocab.add('<PAD>')
        targets = set(tar_temp)

        PyTorchDataset.word_to_num = dict(zip(PyTorchDataset.vocab, range(len(PyTorchDataset.vocab))))
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

    def __init__(self, path):
        self.sentences_var = []
        super(CharDataset, self).__init__(path)

    def load_data(self, path):
        with open(path) as file:
            temp_sentences = []
            temp_targets = []
            sentences = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences.append(list(word)), temp_targets.append(label)
                except ValueError:
                    self.sentences_var.append(temp_sentences)
                    sequens_sen = ' '.join(set([c for row in temp_sentences for c in row]))
                    sequens_tar = ' '.join(temp_targets)
                    sentences.append(sequens_sen)
                    targets.append(sequens_tar)

                    temp_sentences = []
                    temp_targets = []

        return sentences, targets

    def convert2num(self, sentences, targets):
        num_sentences = []
        for sen in self.sentences_var:
            sen_temp = []
            for char_lst in sen:
                sen_temp.append([PyTorchDataset.word_to_num.get(ch, PyTorchDataset.word_to_num['UNK']) for ch in char_lst])
            num_sentences.append(sen_temp)

        num_targets = []
        for tar in targets:
            tar_temp = []
            target = tar.split()
            for ttt in target:
                tar_temp.append(PyTorchDataset.target_to_num.get(ttt))
            num_targets.append(tar_temp)

        return num_sentences, num_targets


class PyTorchDataset_C(PyTorchDataset):

    def __init__(self, path):
        self.sentences_var = []
        super(PyTorchDataset_C, self).__init__(path)

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

    def convert2num(self, sentences, targets):
        num_sentences = []
        for sen in self.sentences_var:
            sen_temp = []
            for word, pref, suff in sen:
                sen_temp.append([PyTorchDataset.word_to_num.get(word, PyTorchDataset.word_to_num['UNK']),
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
