import glob
import os
import numpy as np

source = '/data0/Data/ISIC2019/Data'
dist = '/data0/Data/ISIC2019-openset-refold2/Data'

def split():
  if not os.path.exists(dist):
    os.mkdir(dist)
  if not os.path.exists(dist+'/Train'):
    os.mkdir(dist+'/Train')
  if not os.path.exists(dist+'/Val'):
    os.mkdir(dist+'/Val')
  folders = glob.glob(source+'/*')
  for c in folders:
    Train = c.replace(source, dist+'/Train')
    Val = c.replace(source, dist+'/Val')
    if not os.path.exists(Train):
      os.mkdir(Train)
    if not os.path.exists(Val):
      os.mkdir(Val)
    imgs = glob.glob(c+'/*')
    count = len(imgs)
    imgs = np.array(imgs)
    imgs_val = np.random.choice(imgs, int(count/5))
    imgs_train = np.setdiff1d(imgs, imgs_val)
    for i in imgs_val:
      os.popen('cp "{}" "{}"'.format(i, i.replace(c, Val)))
      print('copy "{}" to "{}"!'.format(i, i.replace(c, Val)))
    for i in imgs_train:
      os.popen('cp "{}" "{}"'.format(i, i.replace(c, Train)))
      print('copy "{}" to "{}"!'.format(i, i.replace(c, Train)))

split()