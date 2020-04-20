import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNNTextClassifier(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, vocab, MODEL_session):
        super(DPCNNTextClassifier, self).__init__()
        embedding_size = int(MODEL_session['embedding_size'])
        update_embedding = MODEL_session.get('update_embedding', False)
        input_dropout_p = float(MODEL_session.get('input_dropout_p', 0.0))
        n_label = int(MODEL_session['n_label'])
        vocab_size = len(vocab)
        embedding = vocab.vectors

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embedding_size), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, n_label)

    def forward(self, input_var, lengths=None):
        embeded = self.embedding(input_var)
        embeded = self.input_dropout(embeded)
        batch, width, height = embeded.shape
        embeded = embeded.view((batch, 1, width, height))
        #print('embeded', embeded.shape)

        # Region embedding
        x = self.conv_region_embedding(embeded)
        #print('conv_region_embedding', x.shape)

        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        #print('x before residual', x.shape)

        cnt = 0
        while x.size()[-2] >= 2:
            x = self._block(x)

            cnt += 1
            #print('x', cnt, x.shape)

        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)

        '''
        embeded torch.Size([99, 1, 129, 300])
        conv_region_embedding torch.Size([99, 250, 127, 1])
        x before residual torch.Size([99, 250, 127, 1])
        x 1 torch.Size([99, 250, 63, 1])
        x 2 torch.Size([99, 250, 31, 1])
        x 3 torch.Size([99, 250, 15, 1])
        x 4 torch.Size([99, 250, 7, 1])
        x 5 torch.Size([99, 250, 3, 1])
        x 6 torch.Size([99, 250, 1, 1])
        '''

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x
