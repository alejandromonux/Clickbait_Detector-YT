from torch import nn

#DEPRECATED
#This was used during the first attempt, then dropped. Kept for legacy purposes.
class Union_ARCH(nn.Module):
    def __init__(self, ):
        super(Union_ARCH, self).__init__()

        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(5, 2) #Forest,XGBoost, Sentiment
        self.OtputFunction = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.OtputFunction(x)