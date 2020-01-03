import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Part3.transducer1 import BidirectionRnn
from Part3.transducer2 import BidirectionRnnCharToSequence


class ComplexRNN(nn.Module):
    """
        BidirectionRnn tagging model.
        representation: regular embedding
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, batch_size, device, bidirectional=True, padding_idx=None):
        """
        Initialize the model
        """
        super(ComplexRNN, self).__init__()
        # Linear layer
        self.modelA = BidirectionRnn(vocab_size=vocab_size,
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     tagset_size=tagset_size,
                                     batch_size=batch_size,
                                     device=device,
                                     padding_idx=padding_idx).to(device)

        self.modelB = BidirectionRnnCharToSequence(vocab_size=vocab_size,
                                                   embedding_dim=embedding_dim,
                                                   hidden_dim=hidden_dim,
                                                   tagset_size=tagset_size,
                                                   batch_size=batch_size,
                                                   device=device,
                                                   padding_idx=padding_idx).to(device)
        self.hidden2out = nn.Linear(tagset_size * 2, tagset_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, sentence, sen_len, word_len):
        """
        The process of the model prediction
        """
        out1 = self.modelA(sentence)
        out2 = self.modelB(sentence, sen_len, word_len)
        out = torch.cat((out1, out2), dim=1)
        out = self.hidden2out(out)
        probs = self.softmax(out)
        return probs.view(sentence.shape[0], sentence.shape[1], -1)
