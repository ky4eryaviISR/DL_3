from torch import nn, Tensor, stack, cat, optim, LongTensor
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class Acceptor(nn.Module):
    def __init__(self, output_shape, emb_length, hidden_1, hidden_2, emb_vec):
        super().__init__()
        self.emb_vec_size = emb_vec
        self.embedded = nn.Embedding(emb_length, emb_vec).to(device)
        self.lstm = nn.LSTM(emb_vec, hidden_1)
        self.hidden = nn.Linear(hidden_1, hidden_2)
        self.output = nn.Linear(hidden_2, output_shape)

        # Define tanh activation and softmax output
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedded(x).view(-1, self.emb_vec_size)
        x, _ = self.lstm(x)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.drop_1(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


parser = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'num': 4
}


def convert_to_numeric(l):
    return [parser.get(word, 4) for word in l]


def load_txt():
    good = [LongTensor(convert_to_numeric(line)) for line in open('pos_examples').read().split('\n')]
    bad = [LongTensor(convert_to_numeric(line) ) for line in open('neg_examples').read().split('\n')]
    total = good + bad
    label = LongTensor([1]*len(good)+[0]*len(bad))
    return total, label


def train(model, data, label, lr=0.01, epoch=10):
    # Sets model to TRAIN mode
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    for t in range(epoch):
        for x, y in zip(data, label):
            x = x.to(device)
            y = y.to(device)
            # Makes predictions
            yhat = model([x])
            # Computes loss
            loss = criterion(yhat, y)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            # Returns the loss
            loss_list.append(loss.item())


acceptor = Acceptor(2, len(parser), 5, 5, 50).to(device)
data, label = load_txt()
train(acceptor,data,label)
print('x')
