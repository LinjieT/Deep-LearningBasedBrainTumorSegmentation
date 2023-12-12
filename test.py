import os
#os.chdir('/content/drive/MyDrive/UNet/Pytorch-UNet-master')
import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import glob
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img[:,0:3,:,:]
    #print(img.size())
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def ComputeIoU(predict, GT):
    intersection = np.logical_and(predict, GT)
    union = np.logical_or(predict, GT)
    if np.sum(union)!=0:
      iou = np.sum(intersection) / np.sum(union)
      return iou,0
    if np.sum(predict)==0 and np.sum(GT)==0:
      iou=1
      return iou,1
    iou=0
    return iou,1
def test(in_files_dic='data/imgs/*.png',OutDic='data/result/',GTDic='data/masks/',GTSuffix='_mask',ModelPath='checkpoints/checkpoint_epoch10.pth'):
  '''
  in_files_dic='data/imgs/*.png'
  OutDic='data/result/'
  GTDic='data/masks/'
  GTSuffix='_mask'
  ModelPath='checkpoints/checkpoint_epoch10.pth'
  '''
  if not os.path.exists(OutDic):
    os.mkdir(OutDic)
  net = UNet(n_channels=3, n_classes=2, bilinear=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net.to(device=device)
  net.load_state_dict(torch.load(ModelPath, map_location=device))
  InputImagePath=glob.glob(in_files_dic)
  mIoU=0
  nmIoU=0
  Numabnormal=0
  for InputPath in InputImagePath:
    #print(InputPath)
    img = Image.open(InputPath)
    mask = predict_img(net=net,
            full_img=img,
            scale_factor=0.5,
            out_threshold=0.5,
            device=device)
    #result = mask_to_image(mask)
    #plt.figure()
    #plt.imshow(mask[1], cmap='gray')
    #Find name of input image
    ImgName=InputPath.split('/')[-1]
    ImgName=ImgName.split('.')[0]

    #Load GT
    GTPath=GTDic+ImgName+GTSuffix+'.png'
    #print(GTPath)
    GT = Image.open(GTPath)
    #GT = BasicDataset.preprocess(GT,0.5, is_mask=True)
    GT= np.asarray(GT)
    if len(GT.shape)==3:
      GT=GT[:,:,0]
    IoU,abnormal= ComputeIoU(mask[1], GT)
    if abnormal==0:
      nmIoU+=IoU
    else:
      Numabnormal+=1
    #print(IoU)
    mIoU+=IoU
    #Path to save prodicted mask
    OutputPath=OutDic+ImgName+'_'+str(IoU)+'.png'

    #Save predicted mask
    Img=mask_to_image(mask[1])
    Img.save(OutputPath)
  mIoU=mIoU/len(InputImagePath)
  nmIoU=nmIoU/(len(InputImagePath)-Numabnormal)
  print('mIoU is: ',mIoU)
  #print('nmIoU is: ',nmIoU)

test(in_files_dic='BraTS/test/T1New/*.png',OutDic='BraTSNewOut/benchmark/',GTDic='BraTS/test/Mask/',GTSuffix='_mask',ModelPath='BraTSBenchmark/checkpoint_epoch10.pth')