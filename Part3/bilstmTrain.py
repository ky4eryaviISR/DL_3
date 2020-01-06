from sys import argv
import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Part3.dataloader import PyTorchDataset, pad_collate_sorted, PyTorchDataset_C, CharDataset, CharSentenceDataset
from Part3.transducer1 import BidirectionRnn
from Part3.transducer2 import BidirectionRnnCharToSequence
from Part3.transducer3 import BidirectionRnnPrefSuff
from Part3.transducer4 import ComplexRNN
from datetime import datetime
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
            'hid': (32, 32),
            'emb_dim': (1000, 80),
            'batch_size': 8,
            'lr': 0.0005,
            'btw_rnn': 100
        }
    }
}


def write2file(type_item, item):
    with open(f"{repr_val}_{corpus}_{type_item}", 'w') as file:
        file.write('\n'.join(item))


def evaluate(model, test_loader, corpus, criterion):
    correct = 0
    total = 0
    loss = 0
    tag_pad_token = PyTorchDataset.target_to_num['<PAD>']
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, len_x, lane_size in test_loader:
            labels = y_batch.view(-1)
            if lane_size is not None:
                probs = model(x_batch, len_x, lane_size).view(-1, tag_size)
            else:
                probs = model(x_batch, len_x).view(-1, tag_size)
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
    print(f'{datetime.now()}: Acc:{100 * correct / total:.2f} Loss:{loss/total:}')
    return 100 * correct / total, loss/total


def train(model, train_loader, test_loader, lr, epoch, corpus):
    loss_dev = []
    acc_dev = []
    if corpus == 'ner':
        print("Loss for NER")
        weight = [0.3 if k == 'O' else 1 for k, v in
                  {k: v for k, v in sorted(PyTorchDataset.target_to_num.items(),
                                           key=lambda item: item[1])}.items()]
        criterion = nn.CrossEntropyLoss(torch.FloatTensor(weight).to(device),
                                        ignore_index=PyTorchDataset.target_to_num['<PAD>'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PyTorchDataset.target_to_num['<PAD>'])
    optimizer = Adam(model.parameters(), lr=lr)
    loss_list = []
    for i in range(epoch):
        passed_sen = 0
        print(f"{datetime.now()}:Epoch number: {i+1}")
        for x_batch, y_batch, len_x, lane_size in train_loader:
            passed_sen += int(y_batch.shape[0])
            y_batch = y_batch.view(-1)

            model.train()
            optimizer.zero_grad()
            if lane_size is not None:
                yhat = model(x_batch, len_x, word_len=lane_size).view(-1, tag_size)
            else:
                yhat = model(x_batch, len_x).view(-1, tag_size)
            loss = criterion(yhat, y_batch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if passed_sen > 500:
                passed_sen = passed_sen % 500
                acc, loss = evaluate(model, test_loader, corpus, criterion)
                loss_dev.append(loss)
                acc_dev.append(acc)

    write2file('loss', loss_dev)
    write2file('acc', acc_dev)
    torch.save(model.state_dict())

if __name__ == '__main__':

    dataset_func = variation[repr_val]['loader']
    model = variation[repr_val]['model']
    lr = variation[repr_val][corpus]['lr']
    batch_size = variation[repr_val][corpus]['batch_size']

    train_dataset = dataset_func(train_file, repr_val)
    test_dataset = dataset_func(test_file,repr_val)
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
    if repr_val == 'd':
        args['vocab_size'] = (len(PyTorchDataset.word_to_num), len(CharSentenceDataset.word_to_num))
    print(args)
    bi_rnn = model(**args).to(device)

    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_sorted)
    test_set = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_sorted)
    train(bi_rnn, train_set, test_set, lr=lr, epoch=5, corpus=corpus)
