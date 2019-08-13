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
  'dataset': 'ISIC2019-submit',
  'model-info': {
    'input-size': (456, 456)
  },
  'dataset-info': {
    'num-of-classes': 8,
    'normalization': {
      'mean': [0.5742, 0.5741, 0.5742],
      'std': [0.1183, 0.1181, 0.1183]
    }
  }
}

def get_dataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': False
  }
  base = '{}/{}'.format(os.environ['datadir-base'], INFO['dataset'])

  val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'))

  imgs = np.array(val_dset.imgs)

  return imgs

# * * * * * * * * * * * * * * * * *
# Evaluator
# * * * * * * * * * * * * * * * * *
def evaluate(tb, vb, modelpath):
  device = os.environ['main-device']
  logging.info('Evaluating program start!')
  # threshold = np.arange(0.5, 1.0001, 0.02)
  threshold = 0.72
  iterations = 50
  dist = modelpath+'/dist'

  # Get dataloader
  imgs = get_dataloaders(tb, vb)

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
    _, inds = softmax.max(1)
    prediction = torch.where(entropy_rate<threshold, inds, torch.tensor([-1]).to(device=device))
    return torch.where(prediction == -1, torch.tensor([1.0]).to(device=device), torch.tensor([0.0]).to(device=device))

  def generate_submit_file(mean_softmax, unknown):
    files = imgs[:, 0]
    filenames = [x[x.rfind('/')+1:x.find('.')] for x in files]
    classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC', 'UNK']
    mean_softmax = mean_softmax.cpu().numpy()
    unknown = unknown.cpu().numpy()[:, None]
    unknown_ind = np.where(unknown == 1)[0]
    res = np.hstack((mean_softmax,unknown))
    df = pd.DataFrame(res, filenames, classes)
    if len(unknown_ind) != 0:
      unknown_images = np.array(filenames)[unknown_ind]
      with open(modelpath+'/unknown.txt', 'w') as f:
        for img in unknown_images:
          f.write('%s\n' % img)
      print('Finish generate unknown image list!')

    df.to_csv(modelpath+'/submission.csv')
    with open(modelpath+'/submission.csv', 'r+') as f:
      old = f.read()
      f.seek(0)
      f.write('image')
      f.write(old)

    print('Finish generate submission file!')

  softmax_list = []
  for path in glob.glob(dist+'/*'):
    softmax = load_softmax(path)
    softmax_list.append(softmax)

  print('Finish load softmax!')
  
  mean_softmax = get_mean_softmax(softmax_list)
  unknown = log_mean_results(threshold, mean_softmax, torch.tensor(imgs[:,1].astype(int)).to(device=device))
  generate_submit_file(mean_softmax, unknown)

if __name__ == '__main__':
  args = vars(GetArgParser().parse_args())
  for k in args.keys():
    INFO[k] = args[k]
  writer, logging = config.run(INFO)
  evaluate(args['train_batch_size'], args['val_batch_size'], args['model_path'])
