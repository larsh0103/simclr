import torch
import torch.nn as nn
from torch.optim import Adam
from simclr import SimCLR
import fire
from torch.nn import CrossEntropyLoss
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class LinearHeadModel(nn.Module):
    def __init__(self, simclr_model_dict,num_classes=10):
        super(LinearHeadModel,self).__init__()
        self.num_classes=num_classes
        self.device = self._get_device()
        if simclr_model_dict:
            print("loading feature extractor")
            smclr = SimCLR(out_dim=128)
            smclr.load_state_dict(torch.load(simclr_model_dict, map_location=torch.device(self.device)))
            self.features = smclr.f

            # ## Freeze feature extractor
            # for param in self.features.parameters():
            #     param.requires_grad = False
            

            self.g = nn.Sequential(nn.Linear(512, out_features=self.num_classes, bias=True))

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device    

    def forward(self,x):
        h = self.features(x)
        out = self.g(h)
        return out

def main(simclr_model_dict : str, epochs: int =1, batch_size=512):
    writer = SummaryWriter("linear_checkpoints")
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
    )
    val_transform  = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
    )
    dataset_train = CIFAR10(root="../../../data",download=True,
                            train=True,transform=train_transform)
    dataset_val =  CIFAR10(root="../../../data",download=False,
                            train=False,transform=val_transform)
    
    indices = np.random.choice(len(dataset_train), len(dataset_train), replace=False)
    sampler = SubsetRandomSampler(indices)
    train_loader=DataLoader(dataset =dataset_train, sampler = sampler,batch_size=batch_size)
    val_loader=DataLoader(dataset =dataset_val, batch_size=batch_size, shuffle=False)

    model = LinearHeadModel(simclr_model_dict=simclr_model_dict,num_classes=10)
    model.to(model.device)
    model.features.requires_grad_ = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters
    optimizer = torch.optim.SGD(
    parameters,
    0.1,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
    momentum=0.9,
    weight_decay=0.,
    nesterov=True)

    model.train()
    device = model.device
    for i in range(epochs):
        total_loss = torch.tensor(0.).to(device)
        acc = 0.0
        total_num = 0
        loader = tqdm(train_loader)
        for j, (img,target) in enumerate(loader):
            out = model(img.to(device))
            
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(out,target.to(device))
            total_num +=img.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * img.size(0)
            correct = (torch.argmax(out.to("cpu").data,1) == target.data).float().sum()
            acc += float(100.0*(correct))
            loader.set_description(f"Epoch: {i}, training_loss: {loss}, accuracy :{acc/total_num}")
        print(f"Epoch {i} training loss: {total_loss/total_num} acc : {acc/total_num}")
        writer.add_scalar("train_loss", total_loss/total_num, global_step=i)
        writer.add_scalar("train_acc", acc/total_num, global_step=i)
        val_loss = 0.0
        acc = 0.0
        total_num = 0
        for j, (img,target) in enumerate(tqdm(val_loader,desc=f'validation epoch: {i}')):
            with torch.no_grad():
                model.eval()
                out = model(img.to(device))
                loss =  torch.nn.functional.cross_entropy(out,target.to(device))
                val_loss +=loss.detach().item() * img.size(0)
                total_num += img.size(0)
                correct = (torch.argmax(out.to("cpu").data,1) == target.data).float().sum()
                acc += float(100.0*(correct))
                loader.set_description(f"Epoch: {i}, val_loss: {val_loss}, accuracy :{acc/total_num}")
        print(f"Epoch {i} validation loss: {val_loss/total_num}, acc : {acc/total_num}")
        writer.add_scalar("val_loss", val_loss/total_num, global_step=i)
        writer.add_scalar("val_acc", acc/total_num, global_step=i)
        model.train()




if __name__ == "__main__":
    fire.Fire(main)