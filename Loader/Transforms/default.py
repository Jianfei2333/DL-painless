import torchvision.transforms as T
import numpy as np

def GetTransform(data_normalize, input_size):
  transform = {
      'train': T.Compose([
        T.Resize(tuple([x*(4/3) for x in input_size])), # 放大
        T.RandomResizedCrop(input_size), # 随机裁剪后resize
        T.RandomHorizontalFlip(0.5), # 随机水平翻转
        T.RandomVerticalFlip(0.5), # 随机垂直翻转
        T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
        T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
        T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
        T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
        T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
        T.ToTensor(),
        data_normalize
      ]), 
      'val': T.Compose([
        T.Resize(input_size), # 放大
        T.ToTensor(),
        data_normalize
      ])
    }
  return transform