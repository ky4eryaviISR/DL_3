import json
from sys import argv

import torch
from dataloader import PyTorchDataset, pad_collate_sorted, PyTorchDataset_C, CharDataset, CharSentenceDataset
from torch import cuda
from torch.utils.data import DataLoader
from transducer1 import BidirectionRnn
from transducer2 import BidirectionRnnCharToSequence
from transducer3 import BidirectionRnnPrefSuff
from transducer4 import ComplexRNN

variation = {
    'a': {
        'loader': PyTorchDataset,
        'model': BidirectionRnn,
        'pos': {
            'hid': 128,
            'emb_dim': 256,
            'batch_size': 16,
            'lr': 0.003,
        },
        'ner': {
            'hid': 100,
            'emb_dim': 1000,
            'batch_size': 64,
            'lr': 0.0007,
        }
    },
    'b': {
        'loader': CharDataset,
        'model': BidirectionRnnCharToSequence,
        'pos': {
            'hid': 100,
            'emb_dim': 80,
            'batch_size': 8,
            'lr': 0.003,
            'btw_rnn': 200
        },
        'ner': {
            'hid': 100,
            'emb_dim': 80,
            'batch_size': 4,
            'lr': 0.0007,
            'btw_rnn': 500
        }
    },
    'c': {
        'loader': PyTorchDataset_C,
        'model': BidirectionRnnPrefSuff,
        'pos': {
            'hid': 128,
            'emb_dim': 256,
            'batch_size': 16,
            'lr': 0.0007,
        },
        'ner': {
            'hid': 100,
            'emb_dim': 1000,
            'batch_size': 64,
            'lr': 0.0007,
        }
    },
    'd': {
        'loader': CharSentenceDataset,
        'model': ComplexRNN,
        'pos': {
            'hid':  (128, 100),
            'emb_dim': (256, 80),
            'batch_size': 16,
            'lr': 0.003,
            'btw_rnn': 500
        },
        'ner': {
            'hid': (100, 10),
            'emb_dim': (1000, 80),
            'batch_size': 8,
            'lr': 0.0003,
            'btw_rnn': 500
        }
    }
}


device = 'cuda' if cuda.is_available() else 'cpu'
repr_val = argv[1]
model_file = argv[2]
test_file = argv[3]
corpus = argv[4]
target_dict = argv[5]
word_dict = argv[6]
word_dict_2 = argv[7] if len(argv) == 8 else None

print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


def load_dict(word_dict, target_dict):
    char_dict = None
    if isinstance(word_dict, tuple):
        char_dict = json.load(open(word_dict[1]))
        word_dict = json.load(open(word_dict[0]))
    else:
        word_dict = json.load( open(word_dict))
    target_dict = json.load(open(target_dict))
    return word_dict, char_dict, target_dict, {v: k for k, v in target_dict.items()}

def padding(x,max):
    diff = max - len(x)
    return x + [0]*diff

def predict(model, test):
    model.eval()
    predicted = []
    if repr_val != 'd':
        test = test[0]
    else:
        test = zip(dataset[0], dataset[1])
    with torch.no_grad():
        for x_batch in test:
            if repr_val in ['b', 'd'] is not None:
                if repr_val == 'd':
                    ch_sentence = x_batch[1]
                    len_x = torch.tensor([len(ch_sentence)])
                    lane_size = torch.tensor([len(i) for i in ch_sentence])
                    max_size = max(lane_size).item()
                    ch_sentence = torch.tensor([padding(x, max_size) for x in ch_sentence]).to(device)
                    x_batch = (x_batch[0].unsqueeze(0), ch_sentence.unsqueeze(0))
                    probs = model(x_batch, len_x, lane_size).view(-1, tag_size)
                else:
                    len_x = torch.tensor([len(x_batch)])
                    lane_size = torch.tensor([len(i) for i in x_batch])
                    max_size = max(lane_size).item()
                    x_batch = torch.tensor([padding(x,max_size) for x in x_batch]).to(device)
                    probs = model(x_batch.unsqueeze(0), len_x, lane_size).view(-1, tag_size)
            else:
                probs = model(x_batch.unsqueeze(0), torch.tensor([x_batch.shape[0]])).view(-1, tag_size)
            predicted.append([indx_2_lbl[i.item()] for i in probs.argmax(dim=1)]+[''])
    with open('result', 'w') as fp:
        fp.write('\n'.join([str(val) for lst in predicted for val in lst]))


def load_data(path):
    with open(path) as file:
        temp_sentences = []
        sentences = []
        temp_characters = []
        characters = []
        unk = word_dict['UNK']
        unk_c = char_dict['UNK'] if repr_val == 'd' else None
        if repr_val == 'a':
            for word in file:
                if word != '\n':
                    temp_sentences.append(word_dict.get(word.strip(), unk))
                else:
                    sentences.append(torch.tensor(temp_sentences).to(device))
                    temp_sentences = []
        elif repr_val == 'b':
            for word in file:
                if word != '\n':
                    temp_sentences.append([word_dict.get(ch, unk) for ch in list(word.strip())])
                else:
                    sentences.append(temp_sentences)
                    temp_sentences = []
        elif repr_val == 'c':
            for word in file:
                if word != '\n':
                    word = word.strip()
                    res = [word_dict.get(word, unk),
                           word_dict.get(word[:3], unk),
                           word_dict.get(word[-3:], unk)]
                    temp_sentences.append(res)
                else:
                    sentences.append(torch.tensor(temp_sentences).to(device))
                    temp_sentences = []
        else:
            for word in file:
                if word != '\n':
                    ch = list(word.strip())
                    word = word_dict.get(word.strip())
                    temp_sentences.append(word_dict.get(word, unk))
                    temp_characters.append([char_dict.get(c, unk_c) for c in ch])
                else:
                    sentences.append(torch.tensor(temp_sentences).to(device))
                    characters.append(temp_characters)
                    temp_sentences = []
                    temp_characters = []

    return sentences, characters


if __name__ == '__main__':
    if word_dict_2 is not None:
        word_dict = (word_dict, word_dict_2)
    word_dict, char_dict, lbl_2_indx, indx_2_lbl = load_dict(word_dict, target_dict)
    dataset = load_data(test_file)
    tag_size = len(lbl_2_indx)
    args = {'vocab_size': len(word_dict),
            'embedding_dim': variation[repr_val][corpus]['emb_dim'],
            'hidden_dim': variation[repr_val][corpus]['hid'],
            'tagset_size': tag_size,
            'batch_size': 1,
            'device': device,
            'padding_idx': 0,
            }
    if repr_val in ['b', 'd']:
        args['btw_rnn'] = variation[repr_val][corpus]['btw_rnn']
    if repr_val == 'd':
        args['vocab_size'] = (len(word_dict), len(char_dict))
    print(args)

    model = variation[repr_val]['model'](**args).to(device)
    model.load_state_dict(torch.load(model_file))
    predict(model, dataset)
