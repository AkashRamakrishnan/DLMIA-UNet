# import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.transform import resize
from dataset import *
from transformations import *
from model import *
from torchvision import transforms
from unet import UNet
from trainer import *
import torch.optim as optim
from loss import dice_loss

## Parameters
train_split = 0.8
train_bs = 4
val_bs = 1
test_bs = 1
in_channels = 1
n_classes = 4
data_transforms = transforms.Compose([transforms.Lambda(normalise), 
                                     transforms.Lambda(resize_image),
                                     transforms.Lambda(np_to_tensor)])

target_transforms = transforms.Compose([transforms.Lambda(reshape_mask),
                                     transforms.Lambda(np_to_tensor)])
data_path = 'data/'

## Datasets
train_dataset = SegmentationDataSet(data_path, train=True, transform=data_transforms, target_transform=target_transforms)
test_dataset = SegmentationDataSet(data_path, train=False, transform=data_transforms, target_transform=target_transforms)
train_size = int(train_split * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
print('Train Set: ', train_dataset.__len__())
print('Validation Set: ', val_dataset.__len__())
print('Test Set: ', test_dataset.__len__())

## Dataloader
dataloader_training = DataLoader(dataset=train_dataset,
                                 batch_size=train_bs,
                                 shuffle=True)
dataloader_validation = DataLoader(dataset=val_dataset,
                                 batch_size=val_bs,
                                 shuffle=True) 
dataloader_test = DataLoader(dataset=test_dataset,
                                 batch_size=test_bs,
                                 shuffle=True)

## Build Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
# model = UNet3D(in_channels=in_channels, n_classes=n_classes).to(device)
model = UNet(in_channels=1,
             out_channels=4,
             n_blocks=3,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)

criterion_1 = torch.nn.CrossEntropyLoss()
# criterion_2 = dice_loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

## train model
# trainer = Trainer(model=model,
#                   device=device,
#                   criterion=criterion,
#                   optimizer=optimizer,
#                   training_DataLoader=dataloader_training,
#                   validation_DataLoader=dataloader_test,
#                   lr_scheduler=None,
#                   epochs=10,
#                   epoch=0,
#                   notebook=False)

# training_losses, validation_losses, lr_rates = trainer.run_trainer()

train(model=model, 
      train_loader = dataloader_training,
      val_loader = dataloader_validation,
      optimizer=optimizer,
      scheduler=scheduler,
      device=device,
      load_path=None)

loss = test(model=model,
     test_loader = dataloader_test,
     device=device)

print(1 - loss)