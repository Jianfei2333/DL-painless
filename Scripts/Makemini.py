import glob
import os
import numpy as np

source = '/data0/Data/ISIC2019-openset-expand/Val'
target = '/data0/Data/ISIC2019-openset-expand-mini/Val'

def makemini():
  num_per_class=10
  folders = glob.glob(source+'/*')
  if not os.path.exists(target):
    os.mkdir(target)
  for f in folders:
    classname = f[f.rfind('/')+1:]
    if not os.path.exists(target+'/'+classname):
      os.mkdir(target+'/'+classname)
    imgs = glob.glob(f+'/*')
    if len(imgs) < num_per_class:
      for img in imgs:
        imgname = img[img.rfind('/')+1:]
        dist_from = img
        dist_to = target+'/'+classname+'/'+imgname
        print ('Moving {} to {}!'.format(dist_from, dist_to))
        os.popen('cp "{}" "{}"'.format(dist_from, dist_to))
    else:
      imgs = np.array(imgs)
      imgs_to_copy = np.random.choice(imgs, num_per_class, replace=False)
      for i in imgs_to_copy:
        imgname = i[i.rfind('/')+1:]
        dist_from = i
        dist_to = target+'/'+classname+'/'+imgname
        print ('Moving {} to {}!'.format(dist_from, dist_to))
        os.popen('cp "{}" "{}"'.format(dist_from, dist_to))


makemini()