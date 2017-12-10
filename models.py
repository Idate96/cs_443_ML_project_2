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

    def forward(self, input):
        return self.model(input)

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
