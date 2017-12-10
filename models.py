from torch import nn
from torch import optim

class BCEModel(nn.Module):
    def __init__(self, embedding_dims, learning_rate=10**-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dims, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid())
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        self.optimizer = None
        self.name = "BCE Model"

    def forward(self, input):
        return self.model(input)

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

class LinearBCEModel(nn.Module):
    def __init__(self, embedding_dims, learning_rate=10**-3, l2_reg=10**-3):
        super().__init__()
        self.l2_reg = l2_reg
        self.model = nn.Sequential(
            nn.Linear(embedding_dims, 1),
            nn.Sigmoid())
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        self.optimizer = None
        self.name = (str(type(self).__name__) + "_lr_" + str(learning_rate) + "_embdim_" + str(embedding_dims) +
                        "_l2_" + str(self.l2_reg))


    def forward(self, input):
        return self.model(input)

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)

class SingleLayerBCEModel(nn.Module):
    def __init__(self, embedding_dims, learning_rate=10**-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dims, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid())
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        self.optimizer = None
        self.name = str(self.__name__) + "_" + str(learning_rate) + "_embdim_" + str(embedding_dims)

    def forward(self, input):
        return self.model(input)

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
