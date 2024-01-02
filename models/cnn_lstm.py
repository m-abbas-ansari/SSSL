import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable

class G_LSTM(nn.Module):
    """ 
    LSTM implementation proposed by A. Graves (2013),
    it has more parameters compared to original LSTM
    """

    def __init__(self, input_size=2048, hidden_size=512):
        super(G_LSTM, self).__init__()
        # without batch_norm
        self.input_x = nn.Linear(input_size, hidden_size, bias=True)
        self.forget_x = nn.Linear(input_size, hidden_size, bias=True)
        self.output_x = nn.Linear(input_size, hidden_size, bias=True)
        self.memory_x = nn.Linear(input_size, hidden_size, bias=True)

        self.input_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.forget_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.memory_h = nn.Linear(hidden_size, hidden_size, bias=True)

        self.input_c = nn.Linear(hidden_size, hidden_size, bias=True)
        self.forget_c = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_c = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, state):
        h, c = state
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c))
        g = torch.tanh(self.memory_x(x) + self.memory_h(h))

        next_c = torch.mul(f, c) + torch.mul(i, g)
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c))
        h = torch.mul(o, next_c)
        state = (h, next_c)

        return state
    
class Sal_seq(nn.Module): 
    def __init__(self, 
                 backend="resnet50",
                 seq_len=20, 
                 im_size = (320, 512), 
                 hidden_size=512, 
                 pretrained=False,
                 weights_loc=None,
                 device="cuda"):
        super(Sal_seq, self).__init__()
        self.seq_len = seq_len
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.device = device

        if pretrained:
            network = torch.load(weights_loc).to(device)
            self.backend = network.backend
            self.rnn = network.rnn
        else:
            # defining backend
            if backend == "resnet50":
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.init_resnet(resnet)
                input_size=2048
            elif backend == "resnet18":
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.init_resnet(resnet)
                input_size=512
            self.rnn = G_LSTM(input_size, hidden_size).to(device)
            
        self.decoder = nn.Linear(hidden_size, 2, bias=True).to(device)  
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def init_resnet(self, resnet):
        self.backend = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)

    def init_vgg(self, vgg):
        # self.backend = vgg.features
        self.backend = nn.Sequential(*list(vgg.features.children())[:-2])  # omitting the last Max Pooling

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size).to(self.device)
        c = torch.zeros(batch, self.hidden_size).to(self.device)
        return (Variable(h), Variable(c))

    def process_lengths(self, pads):
        """
        Computing the lengths of sentences in current batchs
		"""
        max_length = pads.size(1)
        lengths = list(max_length - pads.data.sum(1).squeeze())
        return lengths
    
    def get_fix_tokens(self, x, fixs):
        H, W = self.im_size
        _, feat, h, w = x.size()

        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h-1)/H).long()*w + (fixs[:, :, 1]*(w-1)/W).long()

        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)

        return x

    def crop_seq(self, x, lengths):
        """
		Adaptively select the hidden state at the end of sentences
		"""
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        mask = Variable(mask)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, x.size(2))
        
        return x

    def forward(self, img, fixation, padding_mask):
        valid_len = self.process_lengths(padding_mask[:,-self.seq_len:])  # computing valid fixation lengths
        x = self.backend(img)
        batch, feat, h, w = x.size()
        # recurrent loop
        state = self.init_hidden(batch)  # initialize hidden state
        x = self.get_fix_tokens(x, fixation)
        
        output = []
        for i in range(self.seq_len):
            # extract features corresponding to current fixation
            cur_x = x[:, :, i].contiguous()
            # LSTM forward
            state = self.rnn(cur_x, state)
            output.append(state[0].view(batch, 1, self.hidden_size))

        # selecting hidden states from the valid fixations without padding
        output = torch.cat(output, 1)
        output = self.crop_seq(output, valid_len)
        #output = self.softmax(self.decoder(output))
        return output