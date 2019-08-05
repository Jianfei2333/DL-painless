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

# * * * * * * * * * * * * * * * * *
# Define the training info
# * * * * * * * * * * * * * * * * *
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
    },
    # 'known-classes': ['BCC', 'BKL', 'MEL', 'NV', 'VASC']
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
  # train_loader, train4val_loader, val_loader, num_of_images = get_dataloaders(tb, vb)
  weights = (1/num_of_images)/((1/num_of_images).sum().item())
  # weights = (1/num_of_images)/(1/num_of_images + 1/(num_of_images.sum().item()-num_of_images))
  weights = weights.to(device=device)
  
  # ------------------------------------
  # 2. Define model
  model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['dataset-info']['num-of-classes'])
  model = carrier(model)
  
  # ------------------------------------
  # 3. Define optimizer
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  ignite_scheduler = LRScheduler(scheduler)
  
  # ------------------------------------
  # 4. Define metrics

  # class SoftCrossEntropyLoss(nn.Module):
  #   def __init__(self, weight=None):
  #     super(SoftCrossEntropyLoss, self).__init__()
  #     self.class_weights = weight
    
  #   def forward(self, input, target):
  #     softmax = torch.exp(input) / torch.exp(input).sum(1)[:, None]
  #     onehot_labels = to_onehot(target, input.shape[1])
  #     soft_labels = torch.zeros_like(onehot_labels)
  #     soft_labels = torch.where(onehot_labels.cpu() == 1, torch.tensor([0.9]), torch.tensor([0.1/(input.shape[1]-1)])).to(device=device)
  #     if self.class_weights is not None:
  #       # print(soft_labels.shape, softmax.shape)
  #       loss = -torch.sum(torch.log(softmax) * soft_labels * self.class_weights * input.shape[1])
  #     else:
  #       loss = -torch.sum(torch.log(softmax) * soft_labels)
  #     return loss

  class CrossEntropyLoss_forceavg(nn.Module):
    def __init__(self, weight=None):
      super(CrossEntropyLoss_forceavg, self).__init__()
      self.class_weights = weight.to(device=device)

    def forward(self, input, target):
      ce = nn.functional.cross_entropy(input, target, weight=self.class_weights)
      avg = (input ** 2).sum() - (input[range(input.shape[0]), target] ** 2).sum()
      # print(ce, avg/input.shape[0])
      return ce + avg/input.shape[0]

  class EntropyPrediction(metric.Metric):
    def __init__(self, threshold=0.5):
      super(EntropyPrediction, self).__init__()
      self.threshold = threshold
      self.prediction = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
    
    def reset(self):
      # self.threshold = 0.5
      self.prediction = torch.tensor([])
      self.y = torch.tensor([])
      super(EntropyPrediction, self).reset()
    
    def update(self, output):
      y_pred, y = output
      softmax = torch.exp(y_pred) / torch.exp(y_pred).sum(1)[:, None]
      entropy_base = math.log(y_pred.shape[1])
      entropy = (-softmax * torch.log(softmax)).sum(1)/entropy_base
      values, inds = softmax.max(1)
      prediction = torch.where(entropy<self.threshold, inds, torch.tensor([-1]).to(device=device))
      self.prediction = torch.cat((self.prediction.type(torch.LongTensor).to(device=device), torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)))
      self.y = torch.cat((self.y.type(torch.LongTensor).to(device=device), y.to(device=device)))
      # return self.prediction, self.y

    def compute(self):
      return self.prediction, self.y

  train_metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(CrossEntropyLoss_forceavg(weight=weights)),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(INFO['dataset-info']['num-of-classes']), train_loader.dataset.classes)
  }

  val_metrics = {
    'accuracy': MetricsLambda(Labels2Acc, EntropyPrediction(1.0)),
    'precision_recall': MetricsLambda(Labels2PrecisionRecall, EntropyPrediction(1.0), val_loader.dataset.classes),
    'cmatrix': MetricsLambda(Labels2CMatrix, EntropyPrediction(1.0), val_loader.dataset.classes)
  }
  
  # ------------------------------------
  # 5. Create trainer
  trainer = create_supervised_trainer(model, optimizer, CrossEntropyLoss_forceavg(weight=weights), device=device)
  
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

  # Update process bar on each iteration completed.
  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    log_interval = 1
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
      pbar.desc = desc.format(engine.state.output)
      pbar.update(log_interval)

  @trainer.on(Events.EPOCH_STARTED)
  def refresh_pbar(engine):
    print('Finish epoch {}'.format(engine.state.epoch))
    pbar.refresh()
    pbar.n = pbar.last_print_n = 0

  # Compute metrics on train data on each epoch completed.
  cpe = CustomPeriodicEvent(n_epochs=50)
  cpe.attach(trainer)
  @trainer.on(cpe.Events.EPOCHS_50_COMPLETED)
  def log_training_results(engine):
    print ('Checking on training set.')
    train_evaluator.run(train4val_loader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
      Training Results - Epoch: {}
      Avg accuracy: {:.4f}
      Avg loss: {:.4f}
      precision_recall: \n{}
      confusion matrix: \n{}
      """.format(engine.state.epoch,avg_accuracy,avg_loss,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Train Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Loss', {'Train Loss': avg_loss}, engine.state.epoch)
  
  # Compute metrics on val data on each epoch completed.
  cpe = CustomPeriodicEvent(n_epochs=50)
  cpe.attach(trainer)
  @trainer.on(cpe.Events.EPOCHS_50_COMPLETED)
  def log_validation_results(engine):
    pbar.clear()
    print('* - * - * - * - * - * - * - * - * - * - * - * - *')
    print ('Checking on validation set.')
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
      Validating Results - Epoch: {}
      Avg accuracy: {:.4f}
      precision_recall: \n{}
      confusion matrix: \n{}
      """.format(engine.state.epoch,avg_accuracy,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    writer.add_text(os.environ['run-id'], prompt, engine.state.epoch)
    writer.add_scalars('Aggregate/Acc', {'Val Acc': avg_accuracy}, engine.state.epoch)
    writer.add_scalars('Aggregate/Score', {'Val avg precision': precision_recall['data'][0, -1], 'Val avg recall': precision_recall['data'][1, -1]}, engine.state.epoch)

  # Save model ever N epoch.
  save_model_handler = ModelCheckpoint(os.environ['savedir'], '', save_interval=10, n_saved=2)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model_handler, {'model': model})

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
  torch.autograd.set_detect_anomaly(True)
  run(args['train_batch_size'], args['val_batch_size'], args['lr'], args['e'], writer)