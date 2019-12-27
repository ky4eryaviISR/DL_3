from sys import argv

import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# ToDO: change when submit
# from Part3.dataloader import PyTorchDataset, pad_collate
# from utils import to_print, PRINT, save_graph
from DL_3.Part3.dataloader import PyTorchDataset, pad_collate
from DL_3.utils import to_print, PRINT

BATCH_SIZE = 50

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


class BidirectionRnn(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, bidirectional=True):
        """
        Initialize the model
        """
        super(BidirectionRnn, self).__init__()
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize hidden state in RNN
        self.hidden = self.init_hidden(BATCH_SIZE)

        # RNN - BidirectionRnn with 2 layers
        self.num_direction = 2 if bidirectional else 1
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, dropout=0.5)
        # Linear layer
        self.hidden2out = nn.Linear(hidden_dim * 2, tagset_size)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self,batch_size):
        """
        We need to forget the RNN hidden state before dealing with new sequence
        """
        # The axes semantics are (num_layers*num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(4, batch_size, self.hidden_dim).to(device),
                torch.zeros(4, batch_size, self.hidden_dim).to(device))

    def forward(self, sentence):
        """
        The process of the model prediction
        """
        # (1) input layer
        inputs = sentence
        # (2) embedding layer - Embed the sequences
        embeds = self.embedding(inputs)
        # (3) Feed into the RNN
        rnn_out, self.hidden = self.rnn(embeds.view(sentence.shape[1], -1, self.embedding_dim), self.hidden)
        # (4) Linear layer to tag space
        output = self.hidden2out(rnn_out.view(sentence.shape[0], sentence.shape[1], -1))
        # Softmax
        probs = self.softmax(output)
        probs = probs.view(sentence.shape[0], self.tagset_size, -1)
        return probs


def train(model, train_loader, val_loader, lr=0.01, epoch=10, is_ner=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_list = []
    for t in range(epoch):
        model.train()
        for x_batch, y_batch in train_loader:
            model.hidden = (model.hidden[0].detach(),
                                  model.hidden[1].detach())
            model.hidden = model.init_hidden(x_batch.shape[0])
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Makes predictions
            yhat = model(x_batch)
            # Computes loss
            loss = criterion(yhat, y_batch)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            # Returns the loss
            loss_list.append(loss.item())

        acc_dev, loss_dev = evaluate(val_loader, model, criterion, is_ner)
        acc_tr, loss_tr = evaluate(train_loader, model, criterion, is_ner)
        to_print[PRINT.TRAIN_ACC].append(acc_tr)
        to_print[PRINT.TRAIN_LSS].append(loss_tr)
        to_print[PRINT.TEST_ACC].append(acc_dev)
        to_print[PRINT.TEST_LSS].append(loss_dev)
        print(f"Epoch:{t+1} Train Acc:{acc_tr:.2f} Loss:{loss_tr:.8f} "
              f"Acc Dev Acc: {acc_dev:.2f} Loss:{loss_dev:.8f} ")

    save_graph(to_print[PRINT.TRAIN_ACC], to_print[PRINT.TEST_ACC], 'Accuracy')
    save_graph(to_print[PRINT.TRAIN_LSS], to_print[PRINT.TEST_LSS], 'Loss')
    return model


def evaluate(loader, model, criterion, is_ner):
    loss_list = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            model.hidden = model.init_hidden(x_batch.shape[0])
            outputs = model(x_batch)
            loss_list.append(criterion(outputs, y_batch).item())
            if is_ner:
                for y_real, yhat in zip(y_batch, outputs.argmax(axis=1)):
                    y_real, yhat = int(y_real), int(yhat)
                    if PyTorchDataset.num_to_target[y_real] == 'O' and yhat == y_real:
                        continue
                    if yhat == y_real:
                        correct += 1
                    total += 1
            else:
                total += len(y_batch)
                correct += (outputs.argmax(axis=1) == y_batch).sum().item()

    return 100*correct/total, sum(loss_list)/total


if __name__ == '__main__':
    repr = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    corpus = argv[4]
    test_file = argv[5]
    train_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train'
    # train_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train'
    test_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/dev'
    train_dataset = PyTorchDataset(train_file)
    test_dataset = PyTorchDataset(test_file)
    train_set = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    voc_size = len(PyTorchDataset.word_to_num)
    tag_size = len(PyTorchDataset.target_to_num)
    bi_rnn = BidirectionRnn(vocab_size=voc_size,
                            embedding_dim=25,
                            hidden_dim=50,
                            tagset_size=tag_size).to(device)
    test_set = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    train(bi_rnn, train_set, test_set, lr=0.01, epoch=5, is_ner=False)

