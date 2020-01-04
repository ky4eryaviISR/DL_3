import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionRnnPrefSuff(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, batch_size, device, bidirectional=True, padding_idx=None):
        """
        Initialize the model
        """
        super(BidirectionRnnPrefSuff, self).__init__()
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size
        self.device = device
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Initialize hidden state in RNN
        self.hidden = self.init_hidden(batch_size)

        # RNN - BidirectionRnn with 2 layers
        self.num_direction = 2 if bidirectional else 1
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, dropout=0.3)
        # Linear layer
        self.hidden2out = nn.Linear(hidden_dim * 2, tagset_size)
        self.softmax = nn.Softmax(dim=2)

    def init_hidden(self, batch_size):
        """
        We need to forget the RNN hidden state before dealing with new sequence
        """
        # The axes semantics are (num_layers*num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(4, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(4, batch_size, self.hidden_dim).to(self.device))

    def forward(self, sentence, word_len):
        """
        The process of the model prediction
        """
        # (1) input layer
        inputs = sentence
        # (2) embedding layer - Embed the sequences
        embeds = self.embedding(inputs).sum(2)
        embeds = pack_padded_sequence(embeds, word_len, batch_first=True, enforce_sorted=False)
        # (3) Feed into the RNN
        rnn_out, self.hidden = self.rnn(embeds)
        # (4) Linear layer to tag space
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        output = self.hidden2out(rnn_out)
        # Softmax
        probs = self.softmax(output)
        return probs.view(sentence.shape[0], sentence.shape[1], -1)
