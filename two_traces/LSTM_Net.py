'''  LSTM Network  '''

import torch
import torch.nn as nn 
import torch.nn.functional as F 


# Doc for lstm  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
class LSTM_Net(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, 
            output_size=1,batch_sz=32,num_layer=1,activation='linear'):
        super().__init__()
        self.activation= activation
        self.hidden_layer_size = hidden_layer_size
        self.num_layers= num_layer
        self.batch_sz=batch_sz
        dropout=0
        self.lstm = nn.LSTM(input_size, hidden_layer_size,
        num_layers=self.num_layers,batch_first=True,dropout=dropout)

        self.linear1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)


        self.hidden_cell = (torch.zeros(self.num_layers,batch_sz,self.hidden_layer_size).float().cuda(),
                            torch.zeros(self.num_layers,batch_sz,self.hidden_layer_size).float().cuda())


    def forward(self, input_seq):
        out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # out = self.linear1(out)
        if self.activation =='tanh':
            out = torch.tanh(self.linear(out))
        else:
            out = self.linear(out)
        # out = torch.tanh(self.linear(out))
        # out = self.linear2(out)
        return out
    

    def h_init(self): 
        self.hidden_cell = (torch.zeros(self.num_layers,self.batch_sz,self.hidden_layer_size).float().cuda(),
                            torch.zeros(self.num_layers,self.batch_sz,self.hidden_layer_size).float().cuda())


#########################################################################
