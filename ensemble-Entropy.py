# Headers
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import pandas as pd
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
import matplotlib
# matplotlib.use('WebAgg')
# print(matplotlib.rcsetup.all_backends)
import matplotlib.pyplot as plt
# plt.show()

from Utils.Argparser import GetArgParser
from Utils.Template import GetTemplate
from Utils.Metric import PrecisionRecallTable, CMatrixTable, Labels2Acc, Labels2PrecisionRecall, Labels2CMatrix
import Utils.Configuration as config
from Utils.Modelcarrier import carrier
from Utils.Fakedata import get_fakedataloader

INFO = {
  'model': 'Efficientnet-b3',
  'dataset': 'ISIC2019-openset-2',
  'model-info': {
    'input-size': (300, 300)
  },
  'dataset-info': {
    'num-of-classes': 6,
    'normalization': {
      'mean': [0.5742, 0.5741, 0.5742],
      'std': [0.1183, 0.1181, 0.1183]
    }
  }
}

def get_dataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  input_size = INFO['model-info']['input-size']
  base = '{}/{}'.format(os.environ['datadir-base'], INFO['dataset'])

  train_dset = dset.ImageFolder('{}/{}'.format(base, 'Train'))
  val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'))

  labels = torch.from_numpy(np.array(train_dset.imgs)[:, 1].astype(int))
  
  mapping = {}
  for c in train_dset.classes:
    if c in val_dset.classes:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx[c]
    else:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx['UNKNOWN']
  mapping[-1] = val_dset.class_to_idx['UNKNOWN']

  train_len = train_dset.__len__()
  val_len = val_dset.__len__()

  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.SequentialSampler(range(val_len)), **kwargs)

  imgs = np.array(val_dset.imgs)

  return None, None, val_loader, None, mapping, imgs

# * * * * * * * * * * * * * * * * *
# Evaluator
# * * * * * * * * * * * * * * * * *
def evaluate(tb, vb, modelpath):
  device = os.environ['main-device']
  logging.info('Evaluating program start!')
  threshold = np.arange(0.5, 1.0001, 0.02)
  iterations = 50
  dist = modelpath+'/dist'
  if not os.path.exists(dist):
    os.mkdir(dist)
  savepath = '{}/{}.csv'.format(dist, 'test')
  # rates = [0.7, 0.3]
  
  # Get dataloader
  _, _, val_loader, _, mapping, imgs = get_dataloaders(tb, vb)

  # Get Model

  def load_softmax(path):
    softmax = np.genfromtxt(path, delimiter=',', dtype=float)
    softmax = torch.tensor(softmax).type(torch.float).to(device=device)
    return softmax

  def get_mean_softmax(l):
    res_softmax = None
    for m in l:
      if res_softmax is None:
        res_softmax = m
      else:
        res_softmax += m
    return res_softmax / len(l)

  def log_mean_results(threshold, softmax, y_true):
    entropy_base = math.log(softmax.shape[1])
    entropy_rate = (-softmax * torch.log(softmax)).sum(1)/entropy_base
    # print(entropy_rate)
    _, inds = softmax.max(1)
    prediction = torch.where(entropy_rate<threshold, inds, torch.tensor([-1]).to(device=device))
    prediction = torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)
    
    avg_accuracy = Labels2Acc((prediction, y_true))
    precision_recall = Labels2PrecisionRecall((prediction, y_true), val_loader.dataset.classes)
    cmatrix = Labels2CMatrix((prediction, y_true), val_loader.dataset.classes)

    prompt = """
      Threshold: {}

      Avg accuracy: {:.4f}

      precision_recall: \n{}

      confusion matrix: \n{}
      """.format(threshold,avg_accuracy,precision_recall['pretty'],cmatrix['pretty'])
    logging.info('\n'+prompt)
    print (prompt)
    return precision_recall['pretty']

  softmax_list = []
  for path in glob.glob(dist+'/*'):
    softmax = load_softmax(path)
    softmax_list.append(softmax)
    print(path)
    log_mean_results(1.0, softmax, torch.tensor(imgs[:,1].astype(int)).to(device=device))

  mean_softmax = get_mean_softmax(softmax_list)
  met_list = []
  for t in threshold:
    met = log_mean_results(t, mean_softmax, torch.tensor(imgs[:,1].astype(int)).to(device=device))
    met_list.append(met)

  # Draw
  # mean_recall = [x['mean']['Recall'] for x in met_list]
  # plt.plot(threshold, mean_recall, label='Ensembled entropy-base rejection')
  # plt.plot(threshold, np.repeat(mean_recall[-1], len(threshold)), label='Ensembled baseline')
  # plt.legend()
  # plt.xlabel('Threshold')
  # plt.ylabel('Mean recall')
  # plt.grid(True)
  # plt.show()
  # * * * * *

if __name__ == '__main__':
  args = vars(GetArgParser().parse_args())
  for k in args.keys():
    INFO[k] = args[k]
  writer, logging = config.run(INFO)
  evaluate(args['train_batch_size'], args['val_batch_size'], args['model_path'])
