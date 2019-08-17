# Headers
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import math

from efficientnet_pytorch import EfficientNet

import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision, ConfusionMatrix, MetricsLambda, metric
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.handlers import ModelCheckpoint
from ignite.utils import to_onehot
from ignite.contrib.handlers import CustomPeriodicEvent

import os
from tqdm import tqdm
import logging
import json
import glob

from Utils.Argparser import GetArgParser
from Utils.Template import GetTemplate
from Utils.Metric import PrecisionRecallTable, CMatrixTable, Labels2Acc, Labels2PrecisionRecall, Labels2CMatrix
import Utils.Configuration as config
from Utils.Modelcarrier import carrier
from Utils.Fakedata import get_fakedataloader

INFO = {
  'model': 'Efficientnet-b3',
  'dataset': 'ISIC2019-submit',
  'model-info': {
    'input-size': (456, 456)
  },
  'dataset-info': {
    'num-of-classes': 8,
    'normalization': {
      'mean': [0.5742, 0.5741, 0.5742],
      'std': [0.1183, 0.1181, 0.1183]
    },
  }
}

def get_dataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  input_size = INFO['model-info']['input-size']
  base = '{}/{}'.format(os.environ['datadir-base'], INFO['dataset'])
  normalize = T.Normalize(mean=INFO['dataset-info']['normalization']['mean'], std=INFO['dataset-info']['normalization']['std'])
  transform = {
    'val': T.Compose([
      T.Resize(608),
      T.RandomResizedCrop(456),
      # T.RandomCrop(456),
      T.ToTensor(),
      normalize
    ])
  }
  val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'), transform=transform['val'])
  val_len = val_dset.__len__()

  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.SequentialSampler(range(val_len)), **kwargs)

  return None, None, val_loader, None, None, None

# * * * * * * * * * * * * * * * * *
# Evaluator
# * * * * * * * * * * * * * * * * *
def evaluate(tb, vb, modelpath):
  device = os.environ['main-device']
  logging.info('Evaluating program start!')
  iterations = 50
  dist = modelpath+'/dist'
  if not os.path.exists(dist):
    os.mkdir(dist)
  savepath = '{}/{}.csv'.format(dist, 'b5-8')

  model_path = glob.glob(modelpath+'/b5/*')[0]
  model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=8)
  print(model.state_dict()['_fc.bias'])
  model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'].module.state_dict())
  model = model.to(device=device)
  print(model.state_dict()['_fc.bias'])

  class entropy(metric.Metric):
    def __init__(self):
      super(entropy, self).__init__()
      self.entropy_rate = torch.tensor([], dtype=torch.float)
      self.inds = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
      self.softmax = torch.tensor([], dtype=torch.float)
    
    def reset(self):
      self.entropy_rate = torch.tensor([], dtype=torch.float)
      self.inds = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
      self.softmax = torch.tensor([], dtype=torch.float)
      super(entropy, self).reset()
    
    def update(self, output):
      y_pred, y = output
      softmax = torch.exp(y_pred) / torch.exp(y_pred).sum(1)[:, None]
      entropy_base = math.log(y_pred.shape[1])
      entropy_rate = (-softmax * torch.log(softmax)).sum(1)/entropy_base
      _, inds = softmax.max(1)
      self.softmax = torch.cat((self.softmax.to(device=device), softmax)).to(device=device)
      self.entropy_rate = torch.cat((self.entropy_rate.to(device=device), entropy_rate)).to(device=device)
      self.y = torch.cat((self.y.type(torch.LongTensor).to(device=device), y.to(device=device)))
      self.inds = torch.cat((self.inds.type(torch.LongTensor).to(device=device), inds.to(device=device)))

    def compute(self):
      return self.softmax, self.entropy_rate, self.inds, self.y

  val_metrics = {
    'result': entropy()
  }

  metric_list = []
  train_loader, train4val_loader, val_loader, num_of_images, mapping, imgs = get_dataloaders(tb, vb)
  for i in range(iterations):
    _, _, val_loader, _, _, _ = get_dataloaders(tb, vb)
    print('Iteration {}'.format(i))
    
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics['result']
    softmax, er, inds, _ = metrics
    m = {
      'model': model_path,
      'softmax': softmax,
      'er': er,
      'inds': inds,
    }
    metric_list.append(m)
    print('\tFinish 1!')

  def get_mean_softmax(metric_list):
    mean_softmax = None
    k = 0
    for metrics in metric_list:
      softmax = metrics['softmax']
      if mean_softmax is not None:
        mean_softmax = mean_softmax + softmax
        k += 1
      else:
        mean_softmax = softmax
        k += 1
    return mean_softmax / len(metric_list)

  def save_softmax(softmax):
    np_softmax = softmax.cpu().numpy()
    np.savetxt(savepath, np_softmax, delimiter=",")

  scores = {}

  mean_softmax = get_mean_softmax(metric_list)
  save_softmax(mean_softmax)

if __name__ == '__main__':
  args = vars(GetArgParser().parse_args())
  for k in args.keys():
    INFO[k] = args[k]
  writer, logging = config.run(INFO)
  evaluate(args['train_batch_size'], args['val_batch_size'], args['model_path'])
