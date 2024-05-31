import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(
        self, n_fc1, n_fc2, activation, n_layers, input_sz, output_sz, PID, **kwargs
    ):
        super(Net, self).__init__()
        # Unpack the dictionary
        self.args = kwargs
        self.dtype = torch.float
        self.device = torch.device("cuda")
        self.input_size = input_sz
        self.output_sz = output_sz
        self.n_layers = torch.nn.ModuleList()
        self.hs1 = n_fc1
        self.hs2 = n_fc2

        self.hidden1 = torch.nn.Linear(self.input_size, self.hs1, bias=True)
        self.act = activation()
        self.hidden2 = torch.nn.Linear(self.hs1, self.hs2, bias=True)
        for i in range(0, n_layers):
            linear_layer = torch.nn.Linear(self.hs2, self.hs2)
            self.n_layers.append(linear_layer)
        self.output = torch.nn.Linear(self.hs2, self.output_sz, bias=True)

    def forward(self, x):
        x = x.float()
        y = self.act(self.hidden1(x))
        y = self.act(self.hidden2(y))
        out = self.output(y)
        y = F.tanh(out)  # [-1,1]
        return y
