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

from Utils.Argparser import GetArgParser
from Utils.Template import GetTemplate
from Utils.Metric import PrecisionRecallTable, CMatrixTable, Labels2Acc, Labels2PrecisionRecall, Labels2CMatrix
import Utils.Configuration as config
from Utils.Modelcarrier import carrier
from Utils.Fakedata import get_fakedataloader

from Utils.contrib.ls import CrossEntropywithLS

# * * * * * * * * * * * * * * * * *
# Define the training info
# * * * * * * * * * * * * * * * * *
INFO = {
  'model': 'Efficientnet-b4',
  'dataset': 'ISIC2019-openset-final',
  'model-info': {
    'input-size': (380, 380)
  },
  'dataset-info': {
    'num-of-classes': 8,
    'normalization': {
      'mean': [0.57225319,0.57213739,0.57223089],
      'std': [0.11947449,0.11916592,0.11941016]
    }
  }
}

# * * * * * * * * * * * * * * * * *
# Define the dataloader
# * * * * * * * * * * * * * * * * *
def get_dataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  input_size = INFO['model-info']['input-size']
  base = '{}/{}'.format(os.environ['datadir-base'], INFO['dataset'])
  normalize = T.Normalize(mean=INFO['dataset-info']['normalization']['mean'], std=INFO['dataset-info']['normalization']['std'])
  transform = {
    'train': T.Compose([
      # T.Resize(tuple([int(x*(4/3)) for x in input_size])), # 放大
      # T.RandomResizedCrop(input_size), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.RandomCrop((input_size)),
      T.ToTensor(),
      normalize
    ]), 
    'val': T.Compose([
      T.Resize(input_size), # 放大
      T.ToTensor(),
      normalize
    ])
  }
  train_dset = dset.ImageFolder('{}/{}'.format(base, 'Train'), transform=transform['train'])
  train4val_dset = dset.ImageFolder('{}/{}'.format(base, 'Train'), transform=transform['val'])
  # val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'), transform=transform['val'])

  labels = torch.from_numpy(np.array(train_dset.imgs)[:, 1].astype(int))
  num_of_images_by_class = torch.zeros(len(train_dset.classes))
  for i in range(len(train_dset.classes)):
    num_of_images_by_class[i] = torch.where(labels == i, torch.ones_like(labels), torch.zeros_like(labels)).sum().item()

  mapping = {}
  # for c in train_dset.classes:
  #   if c in val_dset.classes:
  #     mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx[c]
  #   else:
  #     mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx['UNKNOWN']
  # mapping[-1] = val_dset.class_to_idx['UNKNOWN']

  train_len = train_dset.__len__()
  # val_len = val_dset.__len__()

  train_loader = DataLoader(train_dset, batch_size=train_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  # val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(val_len)), **kwargs)
  val_loader = None

  return train_loader, train4val_loader, val_loader, num_of_images_by_class, mapping

# * * * * * * * * * * * * * * * * *
# Main loop
#   1. Define dataloader
#   2. Define model
#   3. Define optimizer
#   4. Define metrics
#   5. Create trainer
#   6. Create evaluator
#   7. Create event hooks
# * * * * * * * * * * * * * * * * *
def run(tb, vb, lr, epochs, writer):
  device = os.environ['main-device']
  logging.info('Training program start!')
  logging.info('Configuration:')
  logging.info('\n'+json.dumps(INFO, indent=2))

  # ------------------------------------
  # 1. Define dataloader
  train_loader, train4val_loader, val_loader, num_of_images, mapping = get_dataloaders(tb, vb)
  weights = (1/num_of_images)/((1/num_of_images).sum().item())
  # weights = (1/num_of_images)/(1/num_of_images + 1/(num_of_images.sum().item()-num_of_images))
  weights = weights.to(device=device)
  
  # ------------------------------------
  # 2. Define model
  model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=INFO['dataset-info']['num-of-classes'])
  model = carrier(model)
  
  # ------------------------------------
  # 3. Define optimizer
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  ignite_scheduler = LRScheduler(scheduler)
  
  # ------------------------------------
  # 4. Define metrics

  train_metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(CrossEntropywithLS(weight=weights)),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(INFO['dataset-info']['num-of-classes']), train_loader.dataset.classes)
  }
  # ------------------------------------
  # 5. Create trainer
  trainer = create_supervised_trainer(model, optimizer, CrossEntropywithLS(weight=weights), device=device)
  
  # ------------------------------------
  # 6. Create evaluator
  train_evaluator = create_supervised_evaluator(model, metrics=train_metrics, device=device)

  desc = 'ITERATION - loss: {:.4f}'
  pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0)
  )

  # ------------------------------------
  # 7. Create event hooks

  # Update process bar on each iteration completed.
  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    log_interval = 1
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
      pbar.desc = desc.format(engine.state.output)
      pbar.update(log_interval)

  # Refresh Process bar.
  @trainer.on(Events.EPOCH_COMPLETED)
  def refresh_pbar(engine):
    print ('Epoch {} completed!'.format(engine.state.epoch))
    pbar.refresh()
    pbar.n = pbar.last_print_n = 0

  # Compute metrics on train data on each epoch completed.
  # cpe = CustomPeriodicEvent(n_epochs=50)
  # cpe.attach(trainer)
  # @trainer.on(cpe.Events.EPOCHS_50_COMPLETED)
  def log_training_results(engine):
    pbar.refresh()
    print ('Checking on training set.')
    train_evaluator.run(train4val_loader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
      Id: {}
      Training Results - Epoch: {}
      Avg accuracy: {:.4f}
      Avg loss: {:.4f}
      precision_recall: \n{}
      confusion matrix: \n{}
      """.format(os.environ['run-id'],engine.state.epoch,avg_accuracy,avg_loss,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Train Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Loss', {'Train Loss': avg_loss}, engine.state.epoch)
    # pbar.n = pbar.last_print_n = 0

  # Save model ever N epoch.
  save_model_handler = ModelCheckpoint(os.environ['savedir'], '', save_interval=10, n_saved=2)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model_handler, {'model': model})

  cpe = CustomPeriodicEvent(n_epochs=200)
  cpe.attach(trainer)
  trainer.add_event_handler(cpe.Events.EPOCHS_200_COMPLETED, log_training_results)

  # Update learning-rate due to scheduler.
  trainer.add_event_handler(Events.EPOCH_STARTED, ignite_scheduler)

  # ------------------------------------
  # Run
  trainer.run(train_loader, max_epochs=epochs)
  pbar.close()

# * * * * * * * * * * * * * * * * *
# Program entrance
# * * * * * * * * * * * * * * * * *
if __name__ == '__main__':
  args = vars(GetArgParser().parse_args())
  for k in args.keys():
    INFO[k] = args[k]
  writer, logging = config.run(INFO)
  run(args['train_batch_size'], args['val_batch_size'], args['lr'], args['e'], writer)