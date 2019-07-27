import glob
import os
import numpy as np

source = '/home/huihui/Data/ISIC2018-expand/Train'
target = '/home/huihui/Data/ISIC2018-expand-mini/Train'

def makemini():
  folders = glob.glob(source+'/*')
  for f in folders:
    classname = f[f.rfind('/')+1:]
    if not os.path.exists(target+'/'+classname):
      os.mkdir(target+'/'+classname)
    imgs = glob.glob(f+'/*')
    if len(imgs) < 5:
      for img in imgs:
        imgname = img[img.rfind('/')+1:]
        dist_from = img
        dist_to = target+'/'+classname+'/'+imgname
        print ('Moving {} to {}!'.format(dist_from, dist_to))
        os.popen('cp {} {}'.format(dist_from, dist_to))
    else:
      imgs = np.array(imgs)
      imgs_to_copy = np.random.choice(imgs, 5, replace=False)
      for i in imgs_to_copy:
        imgname = i[i.rfind('/')+1:]
        dist_from = i
        dist_to = target+'/'+classname+'/'+imgname
        print ('Moving {} to {}!'.format(dist_from, dist_to))
        os.popen('cp {} {}'.format(dist_from, dist_to))


makemini()