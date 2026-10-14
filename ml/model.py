import torch
from torch import nn

class ANN(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        blocks=[]
        activation={"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,"elu":nn.ELU}[config.activation]
        for size, dropout in zip(config.hidden_layers,config.dropout):
            blocks.append(nn.Linear(width,size))
            if config.batch_norm:
                blocks.append(nn.BatchNorm1d(size))
            blocks.extend([activation(),nn.Dropout(dropout)])
            width=size
        blocks.append(nn.Linear(width,1))
        self.network=nn.Sequential(*blocks)

    def forward(self,x):
        return self.network(x).squeeze(-1)

def predict(model,x,batch_size=4096):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(torch.as_tensor(x[i:i+batch_size],dtype=torch.float32))) for i in range(0,len(x),batch_size)]).numpy()
