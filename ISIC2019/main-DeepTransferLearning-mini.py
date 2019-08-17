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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, Recall, Precision, ConfusionMatrix, MetricsLambda, metric
from ignite.contrib.handlers.param_scheduler import LRScheduler

import os
from tqdm import tqdm
import logging
import json
from itertools import cycle

from Utils.Argparser import GetArgParser
from Utils.Template import GetTemplate
from Utils.Metric import PrecisionRecallTable, CMatrixTable, Labels2Acc, Labels2PrecisionRecall, Labels2CMatrix
import Utils.Configuration as config
from Utils.Modelcarrier import carrier
from Utils.Fakedata import get_fakedataloader

# * * * * * * * * * * * * * * * * *
# Define the training info
# * * * * * * * * * * * * * * * * *
INFO = {
  'model': 'Efficientnet-b3',
  'dataset': 'ISIC2019-openset-refold-mini',
  'model-info': {
    'input-size': (300, 300)
  },
  'dataset-info': {
    'num-of-classes': 6,
    'normalization': {
      'mean': [0.5721789939624365,0.5720740320330704,0.5721462963466771],
      'std': [0.19069751305853744,0.21423087622553325,0.22522116414142548]
    },
  },
  'supportset-info': {
    'name': 'Natural-mini',
    'num-of-classes': 12
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
  support_dset_train = dset.ImageFolder('{}/{}'.format(os.environ['datadir-base'], INFO['supportset-info']['name']), transform=transform['train'])
  support_dset_val = dset.ImageFolder('{}/{}'.format(os.environ['datadir-base'], INFO['supportset-info']['name']), transform=transform['val'])

  labels = torch.from_numpy(np.array(train_dset.imgs)[:, 1].astype(int))
  num_of_images_by_class = torch.zeros(len(train_dset.classes))
  for i in range(len(train_dset.classes)):
    num_of_images_by_class[i] = torch.where(labels == i, torch.ones_like(labels), torch.zeros_like(labels)).sum().item()

  mapping = {}
  for c in train_dset.classes:
    if c in val_dset.classes:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx[c]
    else:
      mapping[train_dset.class_to_idx[c]] = val_dset.class_to_idx['UNKNOWN']
  mapping[-1] = val_dset.class_to_idx['UNKNOWN']

  train_len = train_dset.__len__()
  val_len = val_dset.__len__()
  support_len = support_dset_train.__len__()

  train_loader = DataLoader(train_dset, batch_size=int(train_batchsize/2), sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(val_len)), **kwargs)
  support_train_loader = DataLoader(support_dset_train, batch_size=int(train_batchsize/2), sampler=sampler.RandomSampler(range(support_len)), **kwargs)
  support_val_loader = DataLoader(support_dset_val, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(support_len)), **kwargs)

  return train_loader, train4val_loader, val_loader, num_of_images_by_class, mapping, support_train_loader, support_val_loader

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
  train_loader, train4val_loader, val_loader, num_of_images, mapping, support_train_loader, support_val_loader = get_dataloaders(tb, vb)
  weights = (1/num_of_images)/((1/num_of_images).sum().item())
  weights = weights.to(device=device)

  # Build iterable mix up batch loader
  it = iter(train_loader)
  sup_it = iter(support_train_loader)
  mixup_batches = zip(it, cycle(sup_it))

  # ------------------------------------
  # 2. Define model
  model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['dataset-info']['num-of-classes'])
  model = carrier(model)
  support_model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['supportset-info']['num-of-classes'])
  support_model = carrier(support_model)
  
  # ------------------------------------
  # 3. Define optimizer
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  ignite_scheduler = LRScheduler(scheduler)

  support_optimizer = optim.SGD(support_model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(support_optimizer, T_max=200)
  support_ignite_scheduler = LRScheduler(scheduler)
  
  # ------------------------------------
  # 4. Define metrics
  train_metrics = {
    'accuracy': Accuracy(),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(INFO['dataset-info']['num-of-classes']), train_loader.dataset.classes)
  }

  support_metrics = {
    'accuracy': Accuracy(),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), support_val_loader.dataset.classes)
  }

  class DeepTransPrediction(metric.Metric):
    def __init__(self, threshold=torch.tensor([0.5]).repeat(len(train_loader.dataset.classes))):
      super(DeepTransPrediction, self).__init__()
      threshold = threshold.to(device=device)
      self.threshold = threshold
      self.prediction = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
    
    def reset(self):
      self.threshold = torch.tensor([0.5]).repeat(len(train_loader.dataset.classes)).to(device=device)
      self.prediction = torch.tensor([])
      self.y = torch.tensor([])
      super(DeepTransPrediction, self).reset()
    
    def update(self, output):
      y_pred, y = output
      softmax = torch.exp(y_pred) / torch.exp(y_pred).sum(1)[:, None]
      values, inds = softmax.max(1)
      prediction = torch.where(values>self.threshold[inds], inds, torch.tensor([-1]).to(device=device))
      self.prediction = torch.cat((self.prediction.type(torch.LongTensor).to(device=device), torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)))
      self.y = torch.cat((self.y.type(torch.LongTensor).to(device=device), y.to(device=device)))
      # return self.prediction, self.y

    def compute(self):
      return self.prediction, self.y

  val_metrics = {
    'accuracy': MetricsLambda(Labels2Acc, DeepTransPrediction()),
    'precision_recall': MetricsLambda(Labels2PrecisionRecall, DeepTransPrediction(), val_loader.dataset.classes),
    'cmatrix': MetricsLambda(Labels2CMatrix, DeepTransPrediction(), val_loader.dataset.classes)
  }
  
  # ------------------------------------
  # 5. Create trainer
  # trainer = create_supervised_trainer(model, optimizer, nn.CrossEntropyLoss(weight=weights), device=device)

  def membership_loss(input, target, weights):
    _lambda = 5
    classes = input.shape[1]
    sigmoid = 1 / (1 + torch.exp(-input))
    part1 = 1-sigmoid[range(sigmoid.shape[0]), target]
    part1 = (part1 * part1 * weights[target]).sum()
    sigmoid[range(sigmoid.shape[0])] = 0
    part2 = (sigmoid * sigmoid * weights).sum()
    return part1 + _lambda*float(1/(classes-1))*part2

  def step(engine, batch):
    model.train()
    support_model.train()

    _alpha1 = 1
    _alpha2 = 1

    known, support = batch
    x_known, y_known = known
    x_support, y_support = support

    x_known = x_known.to(device=device)
    y_known = y_known.to(device=device)
    x_support = x_support.to(device=device)
    y_support = y_support.to(device=device)

    support_scores = support_model(x_support)
    support_cross_entropy = nn.functional.cross_entropy(support_scores, y_support)

    known_scores = model(x_known)
    known_cross_entropy = nn.functional.cross_entropy(known_scores, y_known, weights)
    known_membership = membership_loss(known_scores, y_known, weights)

    loss = support_cross_entropy + known_cross_entropy * _alpha1 + known_membership * _alpha2

    model.zero_grad()
    support_model.zero_grad()

    loss.backward()

    optimizer.step()
    support_optimizer.step()

    return {
      'Rce_loss': support_cross_entropy.item(),
      'Tce_loss': known_cross_entropy.item(),
      'Tm_loss': known_membership.item(),
      'total_loss': loss.item()
    }

  trainer = Engine(step)

  # ------------------------------------
  # 6. Create evaluator
  train_evaluator = create_supervised_evaluator(model, metrics=train_metrics, device=device)
  val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
  support_evaluator = create_supervised_evaluator(support_model, metrics=support_metrics, device=device)

  desc = 'ITERATION - loss: {:.2f}|{:.2f}|{:.2f}|{:.2f}'
  pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0,0,0,0)
  )

  # ------------------------------------
  # 7. Create event hooks

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    log_interval = 1
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
      o = engine.state.output
      pbar.desc = desc.format(o['Rce_loss'], o['Tce_loss'], o['Tm_loss'], o['total_loss'])
      pbar.update(log_interval)

  @trainer.on(Events.EPOCH_STARTED)
  def rebuild_dataloader(engine):
    pbar.clear()
    print('Rebuild dataloader!')
    it = iter(train_loader)
    sup_it = iter(support_train_loader)
    engine.state.dataloader = zip(it, cycle(sup_it))

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    print ('Checking on training set.')
    train_evaluator.run(train4val_loader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
      Id: {}
      Training Results - Epoch: {}
      Avg accuracy: {:.4f}
      
      precision_recall: \n{}
      
      confusion matrix: \n{}
      """.format(os.environ['run-id'], engine.state.epoch,avg_accuracy,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Train Acc': avg_accuracy}, engine.state.epoch)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_support_results(engine):
    pbar.clear()
    print ('* - * - * - * - * - * - * - * - * - * - * - *')
    print ('Checking on support set.')
    support_evaluator.run(support_val_loader)
    metrics = support_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    precision_recall = metrics['precision_recall']

    prompt = """
    Id: {}
    Support set Results - Epoch: {}
    Avg accuracy: {:.4f}
    precision_recall: \n{}
    """.format(os.environ['run-id'], engine.state.epoch, avg_accuracy, precision_recall['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Support/Acc', {'Train Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Support/Score', {'Avg precision': precision_recall['data'][0, -1], 'Avg recall': precision_recall['data'][1, -1]}, engine.state.epoch)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    pbar.clear()
    print ('* - * - * - * - * - * - * - * - * - * - * - *')
    print ('Checking on validation set.')
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    unknown = precision_recall['pretty']['UNKNOWN']
    print(unknown)
    prompt = """
      Id: {}
      Validating Results - Epoch: {}
      Avg accuracy: {:.4f}
      Unknown precision: {:.4f}
      Unknown recall: {:.4f}
      
      precision_recall: \n{}
      
      confusion matrix: \n{}
      """.format(os.environ['run-id'], engine.state.epoch,avg_accuracy, unknown['Precision'], unknown['Recall'],precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Val Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Score', {'Val avg Precision': precision_recall['data'][0, -1], 'Val avg Recall': precision_recall['data'][1, -1]}, engine.state.epoch)
    writer.add_scalars('Unknown/Score', {'Unknown Precision': unknown['Precision'], 'Unknown Recall': unknown['Recall']}, engine.state.epoch)
    pbar.n = pbar.last_print_n = 0

  trainer.add_event_handler(Events.EPOCH_STARTED, ignite_scheduler)
  trainer.add_event_handler(Events.EPOCH_STARTED, support_ignite_scheduler)

  # ------------------------------------
  # Run
  trainer.run(mixup_batches, max_epochs=epochs)
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
