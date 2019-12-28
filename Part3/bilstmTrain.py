from sys import argv

import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
# ToDO: change when submit
from Part3.dataloader import PyTorchDataset, pad_collate, PyTorchDataset_C
from Part3.transducer1 import BidirectionRnn
from Part3.transducer3 import BidirectionRnnPrefSuff
from utils import to_print, PRINT, save_graph
#from DL_3.Part3.dataloader import PyTorchDataset, pad_collate, CharDataset
#from DL_3.utils import to_print, PRINT, save_graph
# from DL_3.Part3.dataloader import PyTorchDataset, pad_collate
# from DL_3.utils import to_print, PRINT

BATCH_SIZE = 25

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


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
0000            accuracy: float, calculate the data accuracy.

    """

    model.eval()
    total = 0
    accuracy = 0
    total_loss = 0
    for i, batch in enumerate(dataloader):
        data, labels, len_data, _ = batch
        model.hidden = model.init_hidden(data.shape[0])
        # Set the data to run on GPU
        data = data.to(device)
        labels = labels.view(-1).to(device)

        # Set the gradients to zero
        model.zero_grad()
        probs = model(data, len_data).view(-1, tag_size)

        #clear padding
        prediction = torch.argmax(probs, dim=1)
        tag_pad_token = PyTorchDataset.target_to_num['<PAD>']
        mask = (labels > tag_pad_token)

        # Calculate the loss
        loss = criterion(probs[mask], labels[mask])
        total_loss += loss.item()

        if is_ner == 'ner':
            acc = ner_accuracy_calculation(prediction, labels, PyTorchDataset.num_to_target)
        else:
            acc = (prediction[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
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
        for x_batch, y_batch, len_x, len_y in train_loader:
            model.train()
            passed_sen += int(x_batch.shape[0])
            model.hidden = model.init_hidden(x_batch.shape[0])
            x_batch = x_batch.to(device)
            y_batch = y_batch.view(-1).to(device)
            # Makes predictions
            yhat = model(x_batch, len_x).view(-1, tag_size)
            # Computes loss
            loss = criterion(yhat, y_batch)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            # Returns the loss
            loss_list.append(loss.item())
            if passed_sen % 500 == 0:
                acc_dev, loss_dev = evaluate(model, val_loader, criterion)
                #acc_tr, loss_tr = evaluate(model, train_loader, criterion)
                    # to_print[PRINT.TRAIN_ACC].append(acc_tr)
                    # to_print[PRINT.TRAIN_LSS].append(loss_tr)
                    # to_print[PRINT.TEST_ACC].append(acc_dev)
                    # to_print[PRINT.TEST_LSS].append(loss_dev)
                print(#f"Epoch:{t+1} Train Acc:{acc_tr:.2f} Loss:{loss_tr:.8f} "
                      f"Acc Dev Acc: {acc_dev:.2f} Loss:{loss_dev:.8f} ")

    save_graph(to_print[PRINT.TRAIN_ACC], to_print[PRINT.TEST_ACC], 'Accuracy')
    save_graph(to_print[PRINT.TRAIN_LSS], to_print[PRINT.TEST_LSS], 'Loss')
    return model


variation = {
    'a': {
        'loader': PyTorchDataset,
        'model': BidirectionRnn,
    },
    'b': {
        #'loader': CharDataset
    },
    'c': {
        'loader': PyTorchDataset_C,
        'model': BidirectionRnnPrefSuff,
    }
}


if __name__ == '__main__':
    repr = argv[1]
    train_file = argv[2]
    #train_file = r"/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/train"
    model_file = argv[3]
    is_ner = True if argv[4] == 'ner' else False
    test_file = argv[5]
    #test_file = r"/home/vova/PycharmProjects/deep_exe3/DL_3/Part3/ner/dev"
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
    bi_rnn = model(vocab_size=voc_size,
                   embedding_dim=50,
                   hidden_dim=50,
                   tagset_size=tag_size,
                   batch_size=BATCH_SIZE,
                   device=device,
                   padding_idx=PyTorchDataset.word_to_num['<PAD>']).to(device)
    test_set = DataLoader(test_dataset, batch_size=100, shuffle=True, collate_fn=pad_collate)
    train(bi_rnn, train_set, test_set, lr=0.01, epoch=5, is_ner=False)

