import os
import torch.nn as nn

def carrier(model):
  if os.environ['main-device'] != 'cpu':
    gpus = os.environ['gpus']
    gpu_list = [int(x) for x in gpus.split(',')]
    model = nn.DataParallel(model, gpu_list)
    model = model.to(device=os.environ['main-device'])
  else:
    model = model.to(device=os.environ['main-device'])
  return model