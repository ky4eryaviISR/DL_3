import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


class BidirectionRnn(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, device, bidirectional=True):
        """
        Initialize the model
        """
        super(BidirectionRnn, self).__init__()
        # GPU device
        self.device = device
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN - BidirectionRnn with 2 layers
        # Initialize hidden state in RNN
        self.hidden = self.init_hidden()
        self.num_direction = 2 if bidirectional else 1
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, dropout=0.5)
        # Linear layer
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)



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
        rnn_out, self.hidden = self.rnn(embeds.view(len(sentence), 1, -1), self.hidden)
        # Linear layer to tag space
        output = self.hidden2tag(rnn_out.view(len(sentence), -1))
        # Softmax
        probs = self.softmax(output, dim=1)
        return probs
