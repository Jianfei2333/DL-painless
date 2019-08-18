import json
import time
import os
import logging
import hashlib
import time
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def GetDeviceSelection(gpus, gpu_number):
  if len(gpus) < gpu_number:
    print('Error: The GPU number assigned is more than total number of GPUs.')
    exit(1)
  sorted_gpus = sorted(gpus, key=lambda x: x['memory.total']-x['memory.used'], reverse=True)
  l = [x['index'] for x in sorted_gpus][:gpu_number]
  selected_gpus = ','.join([str(x) for x in l])
  main_device = 'cuda:{}'.format(l[0])
  print('Select gpus are: {}\nMain gpu: {}'.format(selected_gpus, main_device))
  os.environ['main-device'] = main_device
  torch.cuda.set_device(main_device)
  os.environ['gpus'] = selected_gpus
  return

def GetTBLogger():
  logdir = "{}/{}".format(os.environ['tb-logdir-base'], os.environ['run-id'])
  writer = SummaryWriter(log_dir=logdir)
  return writer

def GetLogger():
  logdir = "{}/{}.log".format(os.environ['logdir-base'], os.environ['run-id'])
  logging.basicConfig(filename=logdir, level=logging.INFO,
                      format='%(asctime)s - %(levelname)s ::> %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',filemode='a')
  return logging

def run(info):
  query = json.load(os.popen('gpustat --json'))
  hostname = os.popen('echo $HOST').readlines()[0].strip()
  start = time.strftime('%m-%d-%Y-%H:%M:%S')
  gpu_number = info['gpus']

  # Mode update
  os.environ['mode'] = 'evaluate' if info['evaluate'] else 'train'

  # Device update
  if gpu_number > 0:
    GetDeviceSelection(query['gpus'], gpu_number)
  else:
    os.environ['main-device'] = 'cpu'

  # Base configuration
  configuration = {
    'tianhe': {
      'tb-logdir-base': '/home/sysu_issjyin_2/jianfei/logs',
      'logdir-base': '/home/sysu_issjyin_2/jianfei/Log/runlog',
      'savedir-base': '/home/sysu_issjyin_2/jianfei/models',
      'datadir-base': '/home/sysu_issjyin_2/jianfei/Data'
    },
    'Lab1': {
      'tb-logdir-base': '/data0/jianfei/tensorboard-log',
      'logdir-base': '/data0/jianfei/Log/runlog',
      'savedir-base': '/data0/jianfei/models',
      'datadir-base': '/data15/jianfei/Data'
    },
    'Lab2': {
      'tb-logdir-base': '/data0/jianfei/tensorboard-log',
      'logdir-base': '/data0/jianfei/Log/runlog',
      'savedir-base': '/data0/jianfei/models',
      'datadir-base': '/data16/jianfei/Data'
    },
    'Labpc': {
      'tb-logdir-base': '/home/huihui/Log/tensorboard-log',
      'logdir-base': '/home/huihui/Log/runlog',
      'savedir-base': '/home/huihui/Models',
      'datadir-base': '/data0/Data'
    },
    'isee': {
      'tb-logdir-base': '/home/jianfei/tensorboard-log',
      'logdir-base': '/home/jianfei/Log/runlog',
      'savedir-base': '/home/jianfei/models',
      'datadir-base': '/data0/jianfei/Data'
    }
  }

  conf = configuration[hostname]
  # Update datapath for isee lab cluster.
  if hostname == 'isee':
    if not os.path.exists(conf['datadir-base']):
      conf['datadir-base'] = '/data/jianfei/Data' if os.path.exists('/data/jianfei/Data') else '/data1/jianfei/Data'

  for k in conf.keys():
    os.environ[k] = conf[k]

  # Generate exclusive run-id
  run_id = '_'.join([info['model'], info['dataset'], start, info['remark']])
  id_code = hashlib.sha512(bytes(run_id, 'utf-8')).hexdigest()[:7]
  run_id = '_'.join([run_id, id_code])
  os.environ['run-id'] = run_id

  # Display settings
  pd.set_option('display.max_rows', 15)
  pd.set_option('display.max_colwidth', 10)

  # Distribution settings
  torch.distributed.init_process_group('nccl', init_method=info['dist_method'], world_size=info['world'], rank=info['rank'])

  # Return
  writer = GetTBLogger()
  logging = GetLogger()
  os.environ['savedir'] = "{}/{}".format(os.environ['savedir-base'], run_id)

  return writer, logging