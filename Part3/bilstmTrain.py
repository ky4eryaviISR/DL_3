from sys import argv

import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
# ToDO: change when submit
from Part3.dataloader import PyTorchDataset, pad_collate, CharDataset
from utils import to_print, PRINT, save_graph
# from DL_3.Part3.dataloader import PyTorchDataset, pad_collate
# from DL_3.utils import to_print, PRINT

BATCH_SIZE = 1

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


class biLSTMTagger_A(nn.Module):
    """
    biLSTM tagging model. representation: regular embedding
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, device):
        """
        Initialize the model
        """
        super(biLSTMTagger_A, self).__init__()
        # GPU device
        self.device = device
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN - biLSTM with 2 layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2, dropout=0.5)
        # Linear layer
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        # Initialize hidden state in RNN
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        We need to forget the RNN hidden state before dealing with new sequence
        """
        # The axes semantics are (num_layers*num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(4, 1, self.hidden_dim).to(self.device),
                torch.zeros(4, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence):
        """
        The process of the model prediction
        """
        # Embed the sequences
        embeds = self.embedding(sentence)
        # Feed into the RNN
        rnn_out, self.hidden = self.rnn(embeds.view(len(sentence),1,-1), self.hidden)
        # Linear layer to tag space
        output = self.hidden2tag(rnn_out.view(len(sentence), -1))
        # Softmax
        probs = F.softmax(output, dim=1)
        return probs


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
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, dropout=0.3)
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
        #probs = probs.view(sentence.shape[0], self.tagset_size, -1)
        return probs


def ner_accuracy_calculation(prediction, labels, encoded_labels):
    """
        special accuracy function to ner.
        the calculation is performed by ignoring the label 'o'

        Args:
        -----
            prediction: the predictions to the data
            labels: the correct labels of the data
            encoded_labels: the labels encoded to numbers

        Returns:
        --------
            accuracy: float, calculate the ner data accuracy.

    """

    mask = ~((prediction == encoded_labels['O']) & (labels == encoded_labels['O']))
    ind = mask.nonzero().squeeze()
    check_predictions, check_labels = prediction[ind], labels[ind]
    accuracy = (check_predictions == check_labels).sum().item() / ind.shape[0]
    return accuracy


def evaluate(model, dataloader, criterion):
    """
        evaluate the model and check the test loss.
        create a text file with the loss of all epochs.
        Args:
        -----
            model: neural network with one hidden layer.
            device: optional , run on GPU
            dataset_train:
            data_loader: instance of data loader obj

        Returns:
        --------
            accuracy: float, calculate the data accuracy.

    """

    model.eval()
    total = 0
    accuracy = 0
    total_loss = 0
    for i, batch in enumerate(dataloader):
        data, labels = batch
        model.hidden = model.init_hidden()
        # Set the data to run on GPU
        data = data[0].to(device)
        labels = labels[0].to(device)

        # Set the gradients to zero
        model.zero_grad()
        probs = model(data)

        # Calculate the loss
        loss = criterion(probs, labels)
        total_loss += loss.item()

        prediction = torch.argmax(probs, dim=1)

        if is_ner == 'ner':
            acc = ner_accuracy_calculation(prediction, labels, PyTorchDataset.num_to_target)
        else:
            acc = (prediction == labels).sum().item()
        total += labels.shape[0]
        accuracy += acc

    # Average accuracy and loss
    accuracy /= total
    total_loss /= total

    with open('tag1_{}_loss'.format(is_ner), 'a+') as file:
        file.write('{}\n'.format(total_loss))
    return accuracy, total_loss


def train(model, train_loader, val_loader, lr=0.01, epoch=10, is_ner=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_list = []
    for t in range(epoch):
        passed_sen = 0
        print("new epoch")
        for x_batch, y_batch in train_loader:
            model.train()
            passed_sen += int(x_batch.shape[0])
            model.hidden = model.init_hidden()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Makes predictions
            yhat = model(x_batch[0])
            # Computes loss
            loss = criterion(yhat, y_batch[0])
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            # Returns the loss
            loss_list.append(loss.item())
            if passed_sen % 500 == 0:
                acc_dev, loss_dev = evaluate(model, val_loader, criterion)
                acc_tr, loss_tr = evaluate(model, train_loader, criterion)
                    # to_print[PRINT.TRAIN_ACC].append(acc_tr)
                    # to_print[PRINT.TRAIN_LSS].append(loss_tr)
                    # to_print[PRINT.TEST_ACC].append(acc_dev)
                    # to_print[PRINT.TEST_LSS].append(loss_dev)
                print(f"Epoch:{t+1} Train Acc:{acc_tr:.2f} Loss:{loss_tr:.8f} "
                      f"Acc Dev Acc: {acc_dev:.2f} Loss:{loss_dev:.8f} ")

    save_graph(to_print[PRINT.TRAIN_ACC], to_print[PRINT.TEST_ACC], 'Accuracy')
    save_graph(to_print[PRINT.TRAIN_LSS], to_print[PRINT.TEST_LSS], 'Loss')
    return model
# def train(device, train, dev, parameters):
#     # Adam optimizer
#     optimizer = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['wd'])
#     print(sys.argv[1])
#
#     for epoch in range(parameters['epochs']):
#         loss = train_epoch(dataloader_train, model, optimizer, device)
#         losses.append(loss)
#         accuracy = evaluation(model, device, dataset_train, dataloader_validation)
#         accuracies.append(accuracy)
#         print("epoch: {}, accuracy: {}".format(epoch + 1, accuracy))
#     return model, dataset_train.encoded_words, dataset_train.encode_num_to_label, losses, accuracies
#
# def train_epoch(dataloader_train, model, optimizer, device):
#     """
#     train the model on part of the data in each iteration.
#
#     Args:
#     -----
#         dataloader_train: instance of data loader obj
#         model: neural network with one hidden layer.
#         optimizer: Adam optimizer
#         device: optional , run on GPU
#
#     Returns:
#     --------
#         loss_average: float, calculate the average loss.
#
#     """
#
#     total_loss = 0
#     criterion = nn.CrossEntropyLoss()
#     for i, batch in enumerate(dataloader_train):
#         data, labels = batch
#
#         data = data.to(device)
#         labels = labels.to(device)
#
#         # Set the gradients to zero
#         model.zero_grad()
#         probs = model(data)
#
#         # Calculate the loss
#         loss = F.cross_entropy(probs, labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#     return total_loss / len(dataloader_train)

# def evaluate(loader, model, criterion, is_ner):
#     loss_list = []
#     correct = 0
#     total = 0
#     model.eval()
#     with torch.no_grad():
#         for x_batch, y_batch in loader:
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             model.hidden = model.init_hidden(x_batch.shape[0])
#             outputs = model(x_batch)
#             loss_list.append(criterion(outputs, y_batch).item())
#             if is_ner:
#                 for y_real, yhat in zip(y_batch.tolist(), outputs.argmax(axis=1).tolist()):
#                     for i in range(len(y_real)):
#                         if PyTorchDataset.num_to_target[y_real[i]] == 'O' and yhat[i] == y_real[i]:
#                             continue
#                         if yhat[i] == y_real[i]:
#                             correct += 1
#                         total += 1
#             else:
#                 total += y_batch.shape[0]*y_batch.shape[1]
#                 correct += (outputs.argmax(axis=1) == y_batch).sum().item()
#
#     return 100*correct/total, sum(loss_list)/total

variation = {
    'a': {
        'loader': PyTorchDataset,
        'model': BidirectionRnn,
    },
    'b': {
        'loader': CharDataset
    }
}



if __name__ == '__main__':
    repr = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    is_ner = True if argv[4] == 'ner' else False
    test_file = argv[5]
    dataset_func = variation[repr]['loader']
    model = variation[repr]['model']

    # train_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train'
    # train_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train'
    # test_file = r'/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/dev'
    train_dataset = dataset_func(train_file)
    test_dataset = dataset_func(test_file)
    train_set = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    voc_size = len(PyTorchDataset.word_to_num)
    tag_size = len(PyTorchDataset.target_to_num)
    # bi_rnn = model(vocab_size=voc_size,
    #                embedding_dim=60,
    #                hidden_dim=100,
    #                tagset_size=tag_size).to(device)
    bi_rnn = biLSTMTagger_A(vocab_size=voc_size, embedding_dim=50, hidden_dim=50, tagset_size=tag_size, device=device).to(device)
    test_set = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=pad_collate)
    train(bi_rnn, train_set, test_set, lr=0.1, epoch=5, is_ner=False)

