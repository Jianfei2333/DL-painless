# Headers
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from prettytable import PrettyTable
from argparse import ArgumentParser

from efficientnet_pytorch import EfficientNet

import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision, ConfusionMatrix, MetricsLambda
from ignite.contrib.handlers.param_scheduler import LRScheduler

import os
from tqdm import tqdm
import logging
import json

from Utils.Argparser import GetArgParser
from Utils.Template import GetTemplate
from Utils.Metric import PrecisionRecallTable, CMatrixTable
import Utils.Configuration as config
from Utils.Modelcarrier import carrier

# * * * * * * * * * * * * * * * * *
# Define the training info
# * * * * * * * * * * * * * * * * *
INFO = {
  'model': 'Efficientnet-b3',
  'dataset': 'ISIC2018-expand',
  'model-info': {
    'input-size': (300, 300)
  },
  'dataset-info': {
    'num-of-classes': 196,
    'normalization': {
      'mean': [0.645949285966695,0.5280427721210771,0.5413824851836213],
      'std': [0.2271920448317491,0.21024010089240586,0.22535982492903597]
    },
    'known-classes': ['BCC', 'BKL', 'MEL', 'NV', 'VASC']
  }
}

# * * * * * * * * * * * * * * * * *
# Define the dataloader
# * * * * * * * * * * * * * * * * *
def get_dataloaders(train_batchsize, val_batchsize):
  """
  Dataloader: ISIC2018-expand.
  """
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  input_size = INFO['model-info']['input-size']
  base = '{}/{}'.format(os.environ['datadir-base'], INFO['dataset'])
  normalize = T.Normalize(mean=INFO['dataset-info']['normalization']['mean'], std=INFO['dataset-info']['normalization']['std'])
  transform = {
    'train': T.Compose([
      T.Resize(tuple([int(x*(4/3)) for x in input_size])), # 放大
      T.RandomResizedCrop(input_size), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
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
  val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'), transform=transform['val'])

  labels = torch.tensor(train_dset.imgs)[:, 1]
  num_of_images_by_class = torch.zeros(len(train_dset.classes))
  for i in range(len(train_dset.classes)):
    num_of_images_by_class[i] = torch.where(labels == str(i), torch.ones_like(labels), torch.zeros_like(labels)).sum().item()

  mapping = {}
  for c in train_dset.classes:
    if c in INFO['dataset-info']['known-classes']:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx[c]
    else:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx['UNKNOWN']

  train_len = train_dset.__len__()
  val_len = val_dset.__len__()

  train_loader = DataLoader(train_dset, batch_size=train_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(val_len)), **kwargs)

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
  
  # ------------------------------------
  # 2. Define model
  model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['dataset-info']['num-of-classes'])
  model = carrier(model)
  
  # ------------------------------------
  # 3. Define optimizer
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.MultiSetpLR(optimizer, milestones=[40, 60, 80, 100, 120], gamma=0.1)
  ignite_scheduler = LRScheduler(scheduler)
  
  # ------------------------------------
  # 4. Define metrics
  train_metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(nn.CrossEntropyLoss(weight=weights)),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(INFO['dataset-info']['num_of_classes']), train_loader.dataset.classes)
  }

  def val_pred_transform(output):
    y_pred, y = output
    y_pred = torch.tensor([mapping[x.item()] for x in y_pred])
    return y_pred, y

  val_metrics = {
    'accuracy': Accuracy(val_pred_transform),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(val_pred_transform), Recall(val_pred_transform), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(INFO['dataset-info']['num_of_classes'], output_transform=val_pred_transform), train_loader.dataset.classes)
  }
  
  # ------------------------------------
  # 5. Create trainer
  trainer = create_supervised_trainer(model, optimizer, nn.CrossEntropyLoss(weight=weights), device=device)
  
  # ------------------------------------
  # 6. Create evaluator
  train_evaluator = create_supervised_evaluator(model, metrics=train_metrics, device=device)
  val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

  desc = 'ITERATION - loss: {:.4f}'
  pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0)
  )


  # ------------------------------------
  # 7. Create event hooks
  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    log_interval = 5
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
      pbar.desc = desc.format(engine.state.output)
      pbar.update(log_interval)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    pbar.refresh()
    print ('Checking on training set.')
    train_evaluator.run(train4val_loader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = GetTemplate('default-log').format('Training',engine.state.epoch,avg_accuracy,avg_loss,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Train Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Loss', {'Train Loss': avg_loss}, engine.state.epoch)
    # writer.add_scalars('Aggregate/Score', {'Train avg precision': precision_recall['data'][0, -1], 'Train avg recall': precision_recall['data'][1, -1]}, engine.state.epoch)
    # pbar.n = pbar.last_print_n = 0
  
  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    print ('Checking on validation set.')
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = GetTemplate('default-log').format('Validating',engine.state.epoch,avg_accuracy,avg_loss,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Val Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Loss', {'Val Loss': avg_loss}, engine.state.epoch)
    writer.add_scalars('Aggregate/Score', {'Val avg precision': precision_recall['data'][0, -1], 'Val avg recall': precision_recall['data'][1, -1]}, engine.state.epoch)
    pbar.n = pbar.last_print_n = 0

  trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

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