from sys import argv
import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Part3_new.dataloader import PyTorchDataset, pad_collate_sorted, PyTorchDataset_C, CharDataset, CharSentenceDataset
from Part3_new.transducer1 import BidirectionRnn
from Part3_new.transducer2 import BidirectionRnnCharToSequence
from Part3_new.transducer3 import BidirectionRnnPrefSuff
from Part3_new.transducer4 import ComplexRNN

device = 'cuda' if cuda.is_available() else 'cpu'
repr_val = argv[1]
train_file = argv[2]
model_file = argv[3]
corpus = argv[4]
test_file = argv[5]

print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))

variation = {
    'a': {
        'loader': PyTorchDataset,
        'model': BidirectionRnn,
        'pos': {
            'hid': 80,
            'emb_dim': 128,
            'batch_size': 4,
            'lr': 0.005,
        },
        'ner': {
            'hid': 64,
            'emb_dim': 128,
            'batch_size': 2,
            'lr': 0.003,
        }
    },
    'b': {
        'loader': CharDataset,
        'model': BidirectionRnnCharToSequence,
        'padd_func': [],
        'lr': 0.01,
        'pos': {
            'hid': 80,
            'emb_dim': 128,
            'batch_size': 4,
            'lr': 0.008,
            'btw_rnn': 32
        },
        'ner': {
            'hid': 32,
            'emb_dim': 50,
            'batch_size': 1,
            'lr': 0.0005,
            'btw_rnn': 100
        }
    },
    'c': {
        'loader': PyTorchDataset_C,
        'model': BidirectionRnnPrefSuff,
        'pos': {
            'hid': 64,
            'emb_dim': 32,
            'batch_size': 8,
            'lr': 0.05
        },
        'ner': {
            'hid': 32,
            'emb_dim': 128,
            'batch_size': 8,
            'lr': 0.003,
        }
    },
    'd': {
        'loader': CharSentenceDataset,
        'model': ComplexRNN,
        'padd_func': [],
        'lr': 0.01,
        'pos': {
            'hid': 80,
            'emb_dim': 128,
            'batch_size': 4,
            'lr': 0.005,
            'btw_rnn': 32
        },
        'ner': {
            'hid': 32,
            'emb_dim': 50,
            'batch_size': 1,
            'lr': 0.0005,
            'btw_rnn': 100
        }
    }
}


def evaluate(model, test_loader, corpus, criterion):
    correct = 0
    total = 0
    loss = 0
    tag_pad_token = PyTorchDataset.target_to_num['<PAD>']
    model.eval()
    with torch.no_grad():
        for data, labels, len_data, _, lane_size in test_loader:
            data = data.to(device)
            labels = labels.view(-1).to(device)
            if lane_size is not None:
                probs = model(data, len_data, lane_size).view(-1, tag_size)
            else:
                probs = model(data, len_data).view(-1, tag_size)
            _, predicted = torch.max(probs.data, 1)

            if corpus == 'pos':
                mask = (labels > tag_pad_token)
            else:
                mask = ((labels > tag_pad_token) &
                        (
                            ~(
                                (predicted == PyTorchDataset.target_to_num['O'])
                                & (labels == PyTorchDataset.target_to_num['O'])
                            )
                         )
                        )
            predicted = predicted[mask]
            labels = labels[mask]
            probs = probs[mask]
            if labels.size(0) == 0:
                continue
            loss += criterion(probs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Acc:{100 * correct / total:.2f} Loss:{loss/total:}')


def train(model, train_loader, test_loader, repr_val, lr, epoch, corpus):
    if corpus == 'ner' and repr_val == 'b':
        weight = [1 if k == 'O' else 0.3 for k, v in PyTorchDataset.target_to_num.items()]
        criterion = nn.CrossEntropyLoss(torch.FloatTensor(weight).to(device), reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=lr)

    loss_list = []
    for i in range(epoch):
        passed_sen = 0
        print(f"Epoch number: {i+1}")
        for x_batch, y_batch, len_x, len_y, lane_size in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.view(-1).to(device)

            model.train()
            optimizer.zero_grad()
            passed_sen += int(x_batch.shape[0])
            if lane_size is not None:
                yhat = model(x_batch, len_x, lane_size).view(-1, tag_size)
            else:
                yhat = model(x_batch, len_x).view(-1, tag_size)
            loss = criterion(yhat, y_batch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if passed_sen > 500:
                passed_sen = passed_sen % 500
                evaluate(model, test_loader, corpus,criterion)


if __name__ == '__main__':

    dataset_func = variation[repr_val]['loader']
    model = variation[repr_val]['model']
    lr = variation[repr_val][corpus]['lr']
    batch_size = variation[repr_val][corpus]['batch_size']


    train_dataset = dataset_func(train_file)
    test_dataset = dataset_func(test_file)
    tag_size = len( PyTorchDataset.target_to_num )
    args = {'vocab_size': len(PyTorchDataset.word_to_num),
            'embedding_dim': variation[repr_val][corpus]['emb_dim'],
            'hidden_dim': variation[repr_val][corpus]['hid'],
            'tagset_size': tag_size,
            'batch_size': variation[repr_val][corpus]['batch_size'],
            'device': device,
            'padding_idx': 0,
            }
    if repr_val in ['b', 'd']:
        args['btw_rnn'] = variation[repr_val][corpus]['btw_rnn']
    print(args)
    bi_rnn = model(**args).to(device)

    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_sorted)
    test_set = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate_sorted)
    train(bi_rnn, train_set, test_set, repr_val, lr=lr, epoch=5, corpus=corpus)