import torch
import torch.nn as nn
from torch.optim import Adam
import fire
from torch.nn import CrossEntropyLoss
from torch.utils.data.sampler import RandomSampler

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from simclr import SimCLR, N_XENT
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F

class Cifar10Simclr(CIFAR10):

    def __getitem__(self,index):
        img, target =self.data[index] , self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack([img1,img2]), target


def main(epochs: int =1, batch_size=128):
    
    loss_func = N_XENT()
    writer = SummaryWriter(log_dir="checkpoints")

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
    dataset_train = Cifar10Simclr(root="../../../data",download=True,
                            train=True,transform=train_transform)
    dataset_val =  Cifar10Simclr(root="../../../data",download=False,
                            train=False,transform=val_transform)
    sampler_train = RandomSampler(data_source=dataset_train)
    sampler_val = RandomSampler(data_source=dataset_val)
    train_loader=DataLoader(dataset =dataset_train, sampler = sampler_train,batch_size=batch_size)
    val_loader=DataLoader(dataset =dataset_val,sampler = sampler_val, batch_size=batch_size)

    model = SimCLR(device="cuda")
    optimizer = torch.optim.SGD(model.parameters(),lr=0.3* (batch_size/256), momentum=0.9)
    model.train()
    device = model.device
    for i in range(1,epochs+1):
        total_loss = torch.tensor(0.).to(device)
        total_num = 0
        for j, (img,target) in enumerate(tqdm(train_loader,desc=f'training epoch: {i}')):
            img = img.view(img.shape[0]*2,img.shape[2],img.shape[3],img.shape[4])
            
            out = model(img.to(device))
            
            optimizer.zero_grad()
            loss = loss_func(out)
            total_num +=img.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * img.size(0)
        print(f"Epoch {i} training loss: {total_loss/total_num}")
        writer.add_scalar("train_loss", total_loss/total_num, global_step=i)
        val_loss = 0.0
        total_num = 0
        for j, (img,target) in enumerate(tqdm(val_loader,desc=f'validation epoch: {i}')):
            with torch.no_grad():
                model.eval()
                img = img.view(img.shape[0]*2,img.shape[2],img.shape[3],img.shape[4])
                out = model(img.to(device))
                loss = loss_func(out)
                val_loss +=loss.detach().item() * img.size(0)
                total_num += img.size(0)
        print(f"Epoch {i} validation loss: {val_loss/total_num}")
        writer.add_scalar("val_loss", total_loss/total_num, global_step=i)
        if i % 10 == 0:
            torch.save(model.state_dict(),os.path.join("checkpoints",f"model-{i}.pt"))
        model.train()




if __name__ == "__main__":
    fire.Fire(main)