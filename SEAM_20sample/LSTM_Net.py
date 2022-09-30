'''  LSTM Network  '''

import torch
import torch.nn as nn 
import torch.nn.functional as F 


# Doc for lstm  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
class LSTM_Net(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1,
                batch_sz=32,num_lstm_layer=1,activation=None,dropout=0,bias=True):
        super().__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.num_layers= num_lstm_layer
        self.batch_sz=batch_sz
        self.activation = activation

        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers=self.num_layers,batch_first=True,dropout=dropout,
                           bias=bias)

        self.linear = nn.Linear(hidden_layer_size, output_size,bias=bias)


        self.hidden_cell = (torch.zeros(self.num_layers,batch_sz,self.hidden_layer_size).double().cuda(),
                            torch.zeros(self.num_layers,batch_sz,self.hidden_layer_size).double().cuda())




    def forward(self, input_seq):
        out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        if self.activation =='tanh':
            out = torch.tanh(self.linear(out))
        else:
            out = self.linear(out)
        return out
    

    def h_init(self): 
        self.hidden_cell = (torch.zeros(self.num_layers,self.batch_sz,self.hidden_layer_size).double().cuda(),
                            torch.zeros(self.num_layers,self.batch_sz,self.hidden_layer_size).double().cuda())


#########################################################################
