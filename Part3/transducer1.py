import torch
from torch import nn


class BidirectionRnn(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, batch_size, device, bidirectional=True):
        """
        Initialize the model
        """
        super(BidirectionRnn, self).__init__()
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        self.device = device
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize hidden state in RNN
        self.hidden = self.init_hidden(batch_size)

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
        return (torch.zeros(4, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(4, batch_size, self.hidden_dim).to(self.device))

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
        output = self.hidden2out(rnn_out.view(sentence.shape[1], -1))
        # Softmax
        probs = self.softmax(output)
        return probs