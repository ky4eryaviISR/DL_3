import enum
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam

from torch import cuda
MAX_SIZE = 50
device = 'cuda' if cuda.is_available() else 'cpu'
print()

class Acceptor(nn.Module):
    def __init__(self, output_shape, emb_length, hidden_1, emb_vec_dim, hidden_2=5):
        super().__init__()
        self.hidden_dim = hidden_1
        self.emb_vec_size = emb_vec_dim

        # embedding input
        self.embedded = nn.Embedding(emb_length, emb_vec_dim)

        # additional input params(layer) into lstm
        self.input_hidden = self.init_hidden()

        self.lstm = nn.LSTM(emb_vec_dim, hidden_1)

        self.hidden = nn.Linear(hidden_1, hidden_2)
        self.tanh = nn.Tanh()

        self.output = nn.Linear(hidden_2, output_shape)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, x):
        out = self.embedded(x).view(x.shape[1], -1, self.emb_vec_size)
        out, self.input_hidden = self.lstm(out, self.input_hidden)
        out = self.hidden(out.view(x.shape[1], 1, -1)[-1])
        out = self.tanh(out)
        out = self.output(out)
        out = self.softmax(out)

        return out


parser = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'num': 4,
}


def convert_to_numeric(l):
    # e.g: abcd454gfdsggfdfhg
    vec = [parser.get(word, 4) for word in l]
    return [parser.get(word, 4) for word in l]


def load_txt():

    good = [torch.LongTensor(convert_to_numeric(line)).unsqueeze(0) for line in open('pos_examples_new').read().split('\n')]
    bad = [torch.LongTensor(convert_to_numeric(line)).unsqueeze(0) for line in open('neg_examples_new').read().split('\n')]
    total = good + bad
    label = torch.LongTensor([[1]]*len(good)+[[0]]*len(bad))

    return total, label

class PRINT(enum.Enum):
    TRAIN_ACC = 1
    TRAIN_LSS = 2
    TEST_LSS = 3
    TEST_ACC = 4

def train_model(model, train, dev, lr=0.01, epoch=30):

    to_print = {PRINT.TEST_ACC: [],
                PRINT.TEST_LSS: [],
                PRINT.TRAIN_ACC:  [],
                PRINT.TRAIN_LSS:  []}
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for t in range(epoch):
        model.train()
        random.shuffle(train)
        for x, y in train:
            model.input_hidden = (model.input_hidden[0].detach(), model.input_hidden[1].detach())
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc_dev, loss_dev = evaluate(dev, model, criterion)
        acc_tr, loss_tr = evaluate(train,model,criterion)

        to_print[PRINT.TRAIN_ACC].append(acc_tr)
        to_print[PRINT.TRAIN_LSS].append(loss_tr)
        to_print[PRINT.TEST_ACC].append(acc_dev)
        to_print[PRINT.TEST_LSS].append(loss_dev)
        print(f"Epoch:{t+1} Train Acc:{acc_tr:.2f} Loss:{loss_tr:.4f}  Acc Dev Acc: {acc_dev:.2f} Loss:{loss_dev:.4f} ")
        if acc_dev == 100:
            print("Stop training reached the maximum accuracy on the train")
            break
    save_graph(to_print[PRINT.TRAIN_ACC],to_print[PRINT.TEST_ACC],'Acccracy')
    save_graph(to_print[PRINT.TRAIN_LSS],to_print[PRINT.TEST_LSS],'Loss')
    return acceptor

def save_graph(train,test,y_axis):
    plt.figure()
    plt.plot(train,color='r', label='train')
    plt.plot(test, color='g', label='test')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.ylabel(y_axis)
    plt.savefig(y_axis+'.png')


def evaluate(loader, model, criterion):
    loss_list = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss_list.append(criterion(outputs, y_batch).item())

            total += len(y_batch)
            correct += (outputs.argmax(axis=1) == y_batch).sum().item()

    return 100*correct/total, sum(loss_list)/total


def split_shuffle_data(percent=0.8):
    combined = list(zip(data, label))
    random.shuffle(combined)
    split = int(len(data)*percent)
    return combined[:split], combined[split:]


if __name__ == '__main__':
    acceptor = Acceptor(2, emb_length=len(parser),
                        emb_vec_dim=len(parser), hidden_1=8,  hidden_2=8).to(device)
    data, label = load_txt()

    train, test = split_shuffle_data()
    acceptor = train_model(acceptor, train, test, epoch=100, lr=0.001)
    possible_words = ['1a9b19c9d9', '1a9b1d99999999999999999999999999999999999999999999999999999999999'
                                    '99999999999999999999999999999999999999999999999999999999999999999'
                                    '9999999999999999999999999999999999999999999999999999999999999c9']
    #input_failures = [torch.LongTensor(convert_to_numeric(word)).unsqueeze(0) for word in possible_words]
    #print(acceptor(input_failures))




