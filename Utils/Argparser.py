from argparse import ArgumentParser

def GetArgParser():
  parser = ArgumentParser()
  parser.add_argument('-tb', '--train-batch-size', type=int, default= 20,
                      help='input batch size for training (default: 20)')
  parser.add_argument('-vb', '--val-batch-size', type=int, default=50,
                      help='input batch size for validation (default: 50)')
  parser.add_argument('-e', type=int, default= 200,
                      help='Number of epochs to train (default: 200)')
  parser.add_argument('-lr', type=float, default= 3e-3,
                      help='Learning rate (default: 3e-3)')
  parser.add_argument('--gpus', type=int, default=1,
                      help='Assign GPU numbers to use (default: 1)')
  parser.add_argument('--remark', type=str, default='debug',
                      help='Remark this run (default: `debug`)')
  return parser