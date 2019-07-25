# Headers
import os
import torch
import numpy as np
# from argparse import ArgumentParser
# from efficientnet_pytorch import EfficientNet
# import torchvision.transforms as T
# import torchvision.datasets as dset
# from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch.nn as nn
from ignite.metrics import Accuracy, Loss, Recall, Precision, ConfusionMatrix, MetricsLambda
from tqdm import tqdm
# from spinners import Spinners
from Config.Argparser import GetParser
# from Loader.Dataloaders.ISIC2018 import GetDataloaders
from prettytable import PrettyTable
from Config.Datainfo import GetDatainfo
from Config.Deviceselector import GetGpuChoice
from Loader.Metrics.metrics import PrecisionRecallTable, CMatrixTable
import importlib

template = """
{} Results - Epoch: {}
Avg accuracy: {:.2f}
Avg loss: {:.2f}
precision_recall: \n{}
confusion matrix: \n{}
"""

def run(tb, vb, lr, epochs, device):
  """
  Need:
    - Model info
    - Data info
  """
  Data = importlib.import_module('Loader.Dataloaders.'+os.environ['dataname'])
  train_loader, train4val_loader, val_loader = Data.GetDataloaders(tb, vb)
  
  Model = importlib.import_module('Loader.Networks.'+os.environ['modelname'])
  model = Model.GetModel(num_classes=len(train_loader.dataset.classes))

  device = 'cuda:0'
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  trainer = create_supervised_trainer(model, optimizer, nn.functional.cross_entropy, device=device)
  metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(nn.functional.cross_entropy),
    'precision_recall': MetricsLambda(PrecisionRecallTable, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(CMatrixTable, ConfusionMatrix(len(train_loader.dataset.classes)), train_loader.dataset.classes)
  }
  evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

  desc = 'ITERATION - loss: {:.2f}'
  pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0)
  )

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1

    if iter % 5 == 0:
      pbar.desc = desc.format(engine.state.output)
      pbar.update(5)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    pbar.refresh()
    print ('Checking on training set.')
    evaluator.run(train4val_loader)
    metrics = evaluator.state.metrics

    prompt = template.format(
      'Training',
      engine.state.epoch,
      metrics['accuracy'],
      metrics['loss'],
      metrics['precision_recall'],
      metrics['cmatrix']
    )
    tqdm.write(prompt)
    pbar.n = pbar.last_print_n = 0
  
  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    print ('Checking on validation set.')
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics

    prompt = template.format(
      'Validating',
      engine.state.epoch,
      metrics['accuracy'],
      metrics['loss'],
      metrics['precision_recall'],
      metrics['cmatrix']
    )
    tqdm.write(prompt)
    pbar.n = pbar.last_print_n = 0

  trainer.run(train_loader, max_epochs=epochs)
  pbar.close()


if __name__ == '__main__':
  args = vars(GetParser().parse_args())

  os.environ['modelname'] = args['model']
  os.environ['dataname'] = args['data']
  run(args['train_batch_size'], args['val_batch_size'], args['lr'], args['e'])