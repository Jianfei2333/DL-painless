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
  'dataset': 'ISIC2019-openset-2',
  'model-info': {
    'input-size': (300, 300)
  },
  'dataset-info': {
    'num-of-classes': 6,
    'normalization': {
      # 'mean': [0.5721789939624365,0.5720740320330704,0.5721462963466771],
      # 'std': [0.19069751305853744,0.21423087622553325,0.22522116414142548]
      'mean': [0.5742, 0.5741, 0.5742],
      'std': [0.1183, 0.1181, 0.1183]
      # 'mean': [0.486, 0.459, 0.408],
      # 'std': [0.229, 0.224, 0.225]
    },
    # 'known-classes': ['BCC', 'BKL', 'MEL', 'NV', 'VASC']
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
      # T.Resize(input_size), # 放大
      # T.CenterCrop(input_size),
      T.Resize(400),
      # T.CenterCrop(300),
      T.RandomResizedCrop(300),
      # T.RandomCrop(300),
      T.ToTensor(),
      normalize
    ])
  }
  train_dset = dset.ImageFolder('{}/{}'.format(base, 'Train'), transform=transform['train'])
  train4val_dset = dset.ImageFolder('{}/{}'.format(base, 'Train'), transform=transform['val'])
  val_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'), transform=transform['val'])
  val_aug_dset = dset.ImageFolder('{}/{}'.format(base, 'Val'), transform=transform['train'])

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
  train4val_loader = DataLoader(train4val_dset, batch_size=val_batchsize, sampler=sampler.SequentialSampler(range(train_len)), **kwargs)
  val_loader = DataLoader(val_dset, batch_size=val_batchsize, sampler=sampler.SequentialSampler(range(val_len)), **kwargs)
  val_aug_loader = DataLoader(val_aug_dset, batch_size=val_batchsize, sampler=sampler.SequentialSampler(range(val_len)), **kwargs)

  imgs = np.array(val_dset.imgs)

  return train_loader, train4val_loader, val_loader, num_of_images_by_class, mapping, imgs
  # return train_loader, train4val_loader, val_aug_loader, num_of_images_by_class, mapping, imgs

# * * * * * * * * * * * * * * * * *
# Evaluator
# * * * * * * * * * * * * * * * * *
def evaluate(tb, vb, modelpath):
  device = os.environ['main-device']
  logging.info('Evaluating program start!')
  threshold = np.arange(0.5, 1.0, 0.02)
  iterations = 50
  dist = modelpath+'/dist'
  if not os.path.exists(dist):
    os.mkdir(dist)
  savepath = '{}/{}.csv'.format(dist, 'b0-4')
  # rates = [0.7, 0.3]
  
  # Get dataloader
  

  # Get Model
  b0_model_paths = glob.glob(modelpath+'/b0/*')
  b1_model_paths = glob.glob(modelpath+'/b1/*')
  b3_model_paths = glob.glob(modelpath+'/b3/*')
  models = []
  model_weights = []
  for modelpath in b3_model_paths:
    # model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['dataset-info']['num-of-classes'])
    # model = carrier(model)
    # model.load_state_dict(torch.load(modelpath, map_location=device))
    if modelpath.find('3.pth') != -1:
      # model = torch.load(modelpath, map_location=device)
      model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=INFO['dataset-info']['num-of-classes'])
      model.load_state_dict(torch.load(modelpath, map_location=device))
      model = carrier(model)
    else:
      model = torch.load(modelpath, map_location=device)['model']
    models.append(model)
    # model_weights.append(rates[0]/len(b3_model_paths))
  
  for modelpath in b1_model_paths:
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=INFO['dataset-info']['num-of-classes'])
    model = carrier(model)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    # model = torch.load(modelpath, map_location=device)['model']
    models.append(model)
    # model_weights.append(rates[1]/len(b0_model_paths))
  
  for modelpath in b0_model_paths:
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=INFO['dataset-info']['num-of-classes'])
    model = carrier(model)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    # model = torch.load(modelpath, map_location=device)['model']
    models.append(model)
    # model_weights.append(rates[1]/len(b0_model_paths))

  model_paths = b3_model_paths
  model_paths.extend(b1_model_paths)
  model_paths.extend(b0_model_paths)

  class entropy(metric.Metric):
    def __init__(self):
      super(entropy, self).__init__()
      # self.values = torch.tensor([], dtype=torch.float)
      self.entropy_rate = torch.tensor([], dtype=torch.float)
      self.inds = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
      self.softmax = torch.tensor([], dtype=torch.float)
    
    def reset(self):
      # self.values = torch.tensor([])
      self.entropy_rate = torch.tensor([], dtype=torch.float)
      self.inds = torch.tensor([], dtype=torch.int)
      self.y = torch.tensor([], dtype=torch.int)
      self.softmax = torch.tensor([], dtype=torch.float)
      super(entropy, self).reset()
    
    def update(self, output):
      y_pred, y = output
      softmax = torch.exp(y_pred) / torch.exp(y_pred).sum(1)[:, None]
      # print(softmax)
      entropy_base = math.log(y_pred.shape[1])
      entropy_rate = (-softmax * torch.log(softmax)).sum(1)/entropy_base
      _, inds = softmax.max(1)
      # prediction = torch.where(entropy>self.threshold, inds, torch.tensor([-1]).to(device=device))
      # self.prediction = torch.cat((self.prediction.type(torch.LongTensor).to(device=device), torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)))
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
    for ind, model in enumerate(models):
      val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
      val_evaluator.run(val_loader)
      metrics = val_evaluator.state.metrics['result']
      softmax, er, inds, y_true = metrics
      m = {
        'model': model_paths[ind],
        'softmax': softmax,
        'er': er,
        'inds': inds,
        'y': y_true
      }
      metric_list.append(m)
      print('\tFinish 1!')

  def log_validation_results(threshold, metric):
    name = metric['model']
    entropy_rate = metric['er']
    inds = metric['inds']
    y = metric['y']
    # print(entropy)
    # print(threshold)
    # print(inds)
    prediction = torch.where(entropy_rate<threshold, inds, torch.tensor([-1]).to(device=device))
    prediction = torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)

    avg_accuracy = Labels2Acc((prediction, y))
    precision_recall = Labels2PrecisionRecall((prediction, y), val_loader.dataset.classes)
    cmatrix = Labels2CMatrix((prediction, y), val_loader.dataset.classes)
    prompt = """
      Model: {}
      Threshold: {}

      Avg accuracy: {:.4f}

      precision_recall: \n{}

      confusion matrix: \n{}
      """.format(name, threshold,avg_accuracy,precision_recall['pretty'],cmatrix['pretty'])
    tqdm.write(prompt)
    logging.info('\n'+prompt)
    return {
      'mean_recall': precision_recall['pretty']['mean']['Recall']
    }

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

  def get_min_er_softmax(metric_list):
    res_softmax = None
    res_er = None
    for metrics in metric_list:
      if res_softmax is not None:
        mask = torch.where(res_er < metrics['er'], torch.tensor(0).to(device=device), torch.tensor(1).to(device=device))
        mask = mask.nonzero()[:,0]
        res_softmax[mask] = metrics['softmax'][mask]
        res_er[mask] = metrics['er'][mask]
      else:
        res_softmax = metrics['softmax']
        res_er = metrics['er']
    return res_softmax

  def save_softmax(softmax):
    np_softmax = softmax.cpu().numpy()
    np.savetxt(savepath, np_softmax, delimiter=",")

  def log_mean_results(threshold, softmax, y_true):
    entropy_base = math.log(softmax.shape[1])
    entropy_rate = (-softmax * torch.log(softmax)).sum(1)/entropy_base
    # print(entropy_rate)
    _, inds = softmax.max(1)
    prediction = torch.where(entropy_rate<threshold, inds, torch.tensor([-1]).to(device=device))
    prediction = torch.tensor([mapping[x.item()] for x in prediction]).to(device=device)

    high_confidence_inds = (entropy_rate<1e-1).nonzero()
    low_confidence_inds = (entropy_rate>threshold).nonzero()
    high_confidence = np.array([{
      'from': int(imgs[x][1]),
      'to': inds[x].item(),
      'img': imgs[x][0],
      'er': entropy_rate[x].item()
    } for x in high_confidence_inds])
    low_confidence = np.array([{
      'from': int(imgs[x][1]),
      'to': -1,
      'img': imgs[x][0],
      'er': entropy_rate[x].item()
    } for x in low_confidence_inds])

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
    return high_confidence, low_confidence

  scores = {}

  # test1 = log_validation_results(1.0)

  for metrics in metric_list:
    score = log_validation_results(1, metrics)


  for t in threshold:
    mean_softmax = get_mean_softmax(metric_list)
    save_softmax(mean_softmax)
    high, low = log_mean_results(t, mean_softmax, metric_list[0]['y'])
    # high, low = log_mean_results(t, get_min_er_softmax(metric_list), metric_list[0]['y'])

  def transduct(datasets, img_pack, rate=0.8):
    for dset_ind in range(datasets):
      class_to_idx = val_loader.dataset.class_to_idx
      classes = val_loader.dataset.classes
      idx_to_classes = {}
      for c in classes:
        idx_to_classes[class_to_idx[c]] = c

      train_base = '{}/{}/Train'.format(os.environ['datadir-base'], INFO['dataset'])
      # source_base = '{}/{}/Val'.format(os.environ['datadir-base'], INFO['dataset'])
      dist_base = '{}/{}-transduct{}'.format(os.environ['datadir-base'], INFO['dataset'], dset_ind)
      if not os.path.exists(dist_base):
        os.mkdir(dist_base)
      dist_base = '{}/Train'.format(dist_base)
      if not os.path.exists(dist_base):
        os.mkdir(dist_base)

      for c in classes:
        if not os.path.exists('{}/{}'.format(dist_base, c)):
          os.mkdir('{}/{}'.format(dist_base, c))

      to_move = np.random.choice(img_pack, int(rate*img_pack.shape[0]))
      for img in to_move:
        source = img['img']
        img_name = source[source.rfind('/')+1:]
        class_name = idx_to_classes[img['to']] if img['to'] != -1 else 'UNKNOWN'
        dist = '{}/{}/{}'.format(dist_base, class_name, img_name)
        print ('Move "{}" to "{}"!'.format(source, dist))
        os.popen('cp "{}" "{}"'.format(source, dist))
  
      # train_imgs = glob.glob('{}/**/*'.format(train_base))
      # for img in train_imgs:
      #   source = img
      #   dist = img.replace(train_base, dist_base)
      #   print('Move "{}" to "{}"!'.format(source, dist))
      #   os.popen('cp "{}" "{}"'.format(source, dist))

  # transduct(2, high, 0.9)
  # transduct(2, low, 0.6)


  # for pack in high:
    # print ('Move {} to class{}(val calss {})'.format(pack['img'], pack['to'], pack['from']))

  # for threshold in np.arange(0.9, 1.0, 0.001):
  #   score = log_validation_results(threshold)
  #   scores[threshold] = score
  #   print('Finish!')

  # import matplotlib.pyplot as plt
  # x = list(scores.keys())
  # mean_recall = [scores[i]['mean_recall'] for i in scores]

  # plt.plot(x, mean_recall, color='#bfd2d5', label='mean_recall')

  # plt.xlabel('Threshold')
  # plt.grid(linestyle='-.')
  # plt.legend()
  
  # plt.show()

if __name__ == '__main__':
  args = vars(GetArgParser().parse_args())
  for k in args.keys():
    INFO[k] = args[k]
  writer, logging = config.run(INFO)
  evaluate(args['train_batch_size'], args['val_batch_size'], args['model_path'])
