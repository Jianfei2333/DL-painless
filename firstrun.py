# Headers
import torch
import numpy as np
from argparse import ArgumentParser
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch.nn as nn
from ignite.metrics import Accuracy, Loss, Recall, Precision, ConfusionMatrix, MetricsLambda
from tqdm import tqdm
# from spinners import Spinners
from prettytable import PrettyTable


def get_dataloaders(train_batchsize, val_batchsize):
  kwargs={
    'num_workers': 20,
    'pin_memory': True
  }
  normalize = T.Normalize(mean=[0.76352127,0.54612797,0.57053038], std=[0.14121186,0.15289281,0.17033405])
  transform = {
    'train': T.Compose([
      T.Resize((400, 400)), # 放大
      T.RandomResizedCrop((300, 300)), # 随机裁剪后resize
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
      T.Resize((300, 300)), # 放大
      T.ToTensor(),
      normalize
    ])
  }
  train_dset = dset.ImageFolder('/data0/Data/ISIC2018/Train', transform=transform['train'])
  train4val_dset = dset.ImageFolder('/data0/Data/ISIC2018/Train', transform=transform['val'])
  val_dset = dset.ImageFolder('/data0/Data/ISIC2018/Val', transform=transform['val'])
  train_len = train_dset.__len__()
  val_len = val_dset.__len__()

  train_loader = DataLoader(train_dset, batch_size=train_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.RandomSampler(range(val_len)), **kwargs)

  return train_loader, train4val_loader, val_loader

def Precision_Recall_table(p, r, cols):
  tab = PrettyTable()
  tab.add_column('Type', ['Precision', 'Recall'])
  for c in range(len(cols)):
    tab.add_column(cols[c], [round(p[c].item(), 2), round(r[c].item(), 2)])
  tab.add_column('Mean', [p.mean().item(), r.mean().item()])
  return tab

def cmatrix_table(cmatrix, cols):
  tab = PrettyTable()
  cmatrix = cmatrix.numpy().astype(int).T
  tab.add_column('gt\\pd', list(cols))
  for c in range(len(cols)):
    tab.add_column(cols[c], cmatrix[c])
  return tab

def run(tb, vb, lr, epochs):
  train_loader, train4val_loader, val_loader = get_dataloaders(tb, vb)
  model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=7)
  device = 'cuda:0'
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  trainer = create_supervised_trainer(model, optimizer, nn.functional.cross_entropy, device=device)
  metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(nn.functional.cross_entropy),
    # 'precision': Precision(),
    # 'recall': Recall(),
    'precision_recall': MetricsLambda(Precision_Recall_table, Precision(), Recall(), train_loader.dataset.classes),
    'cmatrix': MetricsLambda(cmatrix_table, ConfusionMatrix(7), train_loader.dataset.classes)
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
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
Training Results - Epoch: {}
Avg accuracy: {:.2f}
Avg loss: {:.2f}
precision_recall: \n{}
confusion matrix: \n{}
    """.format(
      engine.state.epoch,
      avg_accuracy,
      avg_loss,
      precision_recall,
      cmatrix
    )
    tqdm.write(prompt)
    pbar.n = pbar.last_print_n = 0
  
  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    print ('Checking on validation set.')
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    precision_recall = metrics['precision_recall']
    cmatrix = metrics['cmatrix']
    prompt = """
Validation Results - Epoch: {}
Avg accuracy: {:.2f}
Avg loss: {:.2f}
precision_recall: \n{}
confusion matrix: \n{}
    """.format(
      engine.state.epoch,
      avg_accuracy,
      avg_loss,
      precision_recall,
      cmatrix
    )
    tqdm.write(prompt)
    pbar.n = pbar.last_print_n = 0

  trainer.run(train_loader, max_epochs=epochs)
  pbar.close()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-tb', '--train-batch-size', type=int, default= 20,
                      help='input batch size for training (default: 20)')
  parser.add_argument('-vb', '--val-batch-size', type=int, default=50,
                      help='input batch size for validation (default: 50)')
  parser.add_argument('-e', type=int, default= 200,
                      help='Number of epochs to train (default: 200)')
  parser.add_argument('-lr', type=float, default= 3e-3,
                      help='Learning rate (default: 3e-3)')
  args = vars(parser.parse_args())
  
  run(args['train_batch_size'], args['val_batch_size'], args['lr'], args['e'])