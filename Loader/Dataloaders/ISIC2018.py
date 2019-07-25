import os
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler
from Loader.Transforms.default import GetTransform
from Config.Modelconf import GetModelConf

def GetDataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  base = '/home/sysu_issjyin_2/jianfei/Data'
  normalize = T.Normalize(mean=[0.76352127,0.54612797,0.57053038], std=[0.14121186,0.15289281,0.17033405])
  transform = GetTransform(normalize, input_size=GetModelConf(os.environ['modelname'])['input_size'])

  train_dset = dset.ImageFolder('{}/{}/{}'.format(base, os.environ['dataname'], 'Train'), transform=transform['train'])
  train4val_dset = dset.ImageFolder('{}/{}/{}'.format(base, os.environ['dataname'], 'Train'), transform=transform['val'])
  val_dset = dset.ImageFolder('{}/{}/{}'.format(base, os.environ['dataname'], 'Val'), transform=transform['val'])

  train_len = train_dset.__len__()
  val_len = val_dset.__len__()

  train_loader = DataLoader(train_dset, batch_size=train_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(val_len)), **kwargs)

  return train_loader, train4val_loader, val_loader