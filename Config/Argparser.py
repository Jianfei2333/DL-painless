from argparse import ArgumentParser

def GetParser():
  parser = ArgumentParser()
  parser.add_argument('-tb', '--train-batch-size', type=int, default= 20,
                      help='input batch size for training (default: 20)')
  parser.add_argument('-vb', '--val-batch-size', type=int, default=50,
                      help='input batch size for validation (default: 50)')
  parser.add_argument('-e', type=int, default= 200,
                      help='Number of epochs to train (default: 200)')
  parser.add_argument('-lr', type=float, default= 3e-3,
                      help='Learning rate (default: 3e-3)')
  parser.add_argument('--model', type=str, default='Efficientnetb3',
                      help='Specify model to use (default: `Efficientnetb3`)')
  parser.add_argument('--data', type=str, default='ISIC2018',
                      help='Specify dataset to use (default: `ISIC2018`)')
  parser.add_argument('--damp', type=str, default='default',
                      help='Specify learning-rate damp rule (default: `default`)')
  parser.add_argument('--remark', type=str, default='debug',
                      help='Add your own remark to this routine of training (default: `debug`)')
  return parser