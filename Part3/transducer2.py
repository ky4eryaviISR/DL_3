import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionRnnCharToSequence(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, batch_size, device, bidirectional=True, padding_idx=None, btw_rnn=10):
        """
        Initialize the model
        """
        super(BidirectionRnnCharToSequence, self).__init__()
        # Dimensions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.btw_rnns = btw_rnn
        self.tagset_size = tagset_size
        self.device = device
        # Representation
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Initialize hidden state in RNN
        self.hidden = self.init_hidden(batch_size)

        # RNN - BidirectionRnn with 2 layers
        self.num_direction = 2 if bidirectional else 1
        self.rnn = nn.LSTM(embedding_dim, self.btw_rnns)
        self.rnn2 = nn.LSTM(self.btw_rnns, hidden_dim, bidirectional=bidirectional, num_layers=2)
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

    def init_hidden_char(self,batch_size):
        # The axes semantics are (num_layers*num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.btw_rnns).to(self.device),
                torch.zeros(1, batch_size, self.btw_rnns).to(self.device))

    def forward(self, sentence, sen_len, word_len,soft_max=True):
        """
        The process of the model prediction
        """
        # (1) input layer
        inputs = sentence.view(sentence.shape[0] * sentence.shape[1], -1)
        batch_word_len = sentence.shape[0]*sentence.shape[1]

        word_len, ord = word_len.sort(dim=0, descending=True)
        inputs = inputs[ord]

        inputs = inputs[word_len > 0]
        word_len = word_len[word_len > 0]
        # (2) embedding layer - Embed the sequences
        embeds = self.embedding(inputs)
        embeds = pack_padded_sequence(embeds, word_len, batch_first=True)
        # (3) Feed into the RNN
        rnn_out, _ = self.rnn(embeds, self.init_hidden_char(inputs.shape[0]))
        # (4) Linear layer to tag space
        rnn_out, lengths = pad_packed_sequence(rnn_out, batch_first=True)
        last_cell = torch.cat([rnn_out[i, j.data - 1]
                               for i, j in enumerate(lengths)])\
            .view(len(word_len), self.btw_rnns)

        last_cell = torch.cat((last_cell,
                               torch.zeros(batch_word_len - len(word_len),
                                           self.btw_rnns,
                                           device=self.device)))
        _, revers = ord.sort(0)
        last_cell = last_cell[revers]
        last_cell = last_cell.view(sentence.shape[0], sentence.shape[1], -1)

        sen_len, ord = sen_len.sort(dim=0, descending=True)
        last_cell = last_cell[ord]

        packed = pack_padded_sequence(last_cell, sen_len, batch_first=True)
        rnn_out2, _ = self.rnn2(packed, self.init_hidden(sentence.shape[0]))
        rnn_out2, _ = pad_packed_sequence(rnn_out2, batch_first=True)

        _, revers = ord.sort(0)
        rnn_out2 = rnn_out2[revers]

        output = self.hidden2out(rnn_out2)
        # Softmax
        if soft_max:
            output = self.softmax(output)
        return output.view(sentence.shape[0], sentence.shape[1], -1)
