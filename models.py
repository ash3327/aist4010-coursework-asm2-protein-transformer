import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, hidden:list[int]=[1000, 256, 256, 15], mode:int=0):
        """
        Note: hidden must be a list of int, where the first and last int denotes the input and output dimensions.
        """
        super(FCNet, self).__init__()

        self.fcs = nn.Sequential()

        for i in range(len(hidden)-1):
            if mode==0:
                self.fcs.append(nn.Linear(hidden[i], hidden[i+1], dtype=float))
            else:
                self.fcs.append(nn.Linear(hidden[i], hidden[i+1]))#, dtype=float))
            self.fcs.append(nn.ReLU(inplace=True))

        self.fcs.pop(-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fcs(x)