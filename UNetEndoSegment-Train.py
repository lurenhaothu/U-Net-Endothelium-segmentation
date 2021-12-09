import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A

import helper


CSV_FILE = './train/df.csv'
DATA_DIR = '/content/'

DEVICE = 'cuda'

EPOCHS = 25
LR = 0.0003
IMG_SIZE = 320
BATCH_SIZE = 1

#ENCODER = 'timm-efficientnet-b0'
#WEIGHTS = 'imagenet'

ENCODER = 'resnet18'
WEIGHTS = 'imagenet'

row = df.iloc[1]

image_path = row.images
mask_path = row.masks

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

train_df, valid_df = train_test_split(df, test_size = 0.9, random_state = 42)

def get_train_augs():
  return A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.RandomSizedCrop(min_max_height=(50, 101), height=IMG_SIZE, width=IMG_SIZE, p=0.5)
  ])

def get_valid_augs():
  return A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE)
  ])

class SegmentationDataset(Dataset):
  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image_path = row.images
    mask_path = row.masks

    #image = cv2.imread(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, axis = -1)
    mask = np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    # (h,w,c) -> (c,h, w)

    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image, mask

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df,get_valid_augs())

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True)

class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()
    self.arc = smp.UnetPlusPlus(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        #encoder_depth = 5,
        decoder_channels = [1025, 512, 256, 128, 64],
        in_channels = 1,
        classes = 1,
        activation = None
    )
  
  def forward(self, images, masks=None):
    logits = self.arc(images)

    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1+loss2
    return logits

model = SegmentationModel()
model.to(DEVICE)

def train_fn(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(data_loader)

def eval_fn(data_loader, model):

  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits, loss = model(images, masks)

      total_loss += loss.item()

  return total_loss / len(data_loader)

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

best_valid_loss = np.Inf

for i in range(100):

  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print("SAVED_MODEL")
    best_valid_loss = valid_loss

  print(f"Epoch : {i+1} Train_loss : {train_loss} Valid_loss : {valid_loss}")
