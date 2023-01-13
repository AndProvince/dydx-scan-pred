import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLSTM(nn.Module):
    def __init__(self, inputDim, hiddenDim, seqLen, outputDim):
        super(ModelLSTM, self).__init__()

        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.lstm1 = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenDim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hiddenDim * seqLen, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=outputDim)

    def forward(self, input, forward=0):
        outputs = []

        x = torch.tanh(self.lstm1(input)[0])
        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        outputs += [x]

        # TODO for next version: redict for next step count (forward)
        '''
        for i in range(forward):
            input = torch.concat([input[:, 1:], x], dim=1)

            x = torch.tanh(self.lstm1(input)[0])
            x = x.reshape(x.shape[0], -1)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            outputs += [x]
        '''
        outputs = torch.stack(outputs, 1).squeeze(1)
        return outputs
