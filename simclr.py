import torch.nn as nn
from torchvision.models import resnet18
import torch

class SimCLR(nn.Module):
    def __init__(self, device = "cpu", out_dim=128, input_shape=(3,32,32)):
        super(SimCLR,self).__init__()
        self.input_shape=input_shape
        self.f = resnet18()
        self.f_out_features = self.f.fc.in_features
        print(self.f_out_features)
        self.out_dim = out_dim
        self.device = device
        print(self.input_shape)
        if self.input_shape[-1]==32:
            self.f.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.f.maxpool = nn.Identity()
        self.f.fc=nn.Identity()
        self.g = nn.Sequential(nn.Linear(in_features=self.f_out_features, out_features=2048), nn.ReLU(),
        nn.Linear(in_features=2048, out_features = self.out_dim))
        self.f.to(device)
        self.g.to(device)

    def forward(self,x):
        h = self.f(x)
        return self.g(h)
    

class N_XENT(nn.Module):

    def forward(self, X, T=0.5):
        X = nn.functional.normalize(X,dim=1)
        sim_matrix = torch.mm(X,X.t())
        sim_matrix = sim_matrix.clamp(min=1e-7) / T

        sim_matrix = sim_matrix - torch.eye(sim_matrix.shape[0],sim_matrix.shape[1]).to(sim_matrix.device) * 1e5

        ## Make array indicating positive samples

        pos = torch.arange(X.shape[0])
        pos[1::2] -=1
        pos[::2] +=1 
        return nn.functional.cross_entropy(input=sim_matrix,target=pos.long().to(sim_matrix.device))