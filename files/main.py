import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import cv2

from data_generator import detection_dataset
from model import Model

if __name__ == '__main__':

    # train only the decoder weights for 50 epochs 
    csv_path = '/home/sbhand2s/downloads/vision_labs_project/files/dataset/Deepansh_Swaroop/labels.csv'
    img_path = '/home/sbhand2s/downloads/vision_labs_project/files/dataset/Deepansh_Swaroop/images/'

    dg = detection_dataset((640,480), img_path, csv_path)
    dataloader = DataLoader(dg, batch_size=20, shuffle=True)

    criterion = nn.MSELoss()
    model = Model()

    for param in list(model.parameters())[:60]:
        param.require_grad = False


    optimizer = Adam(model.parameters())

    model.cuda()
    criterion.cuda()

    epochs = 50

    for epoch in range(epochs):
        epoch_loss = 0
        for data in dataloader:
            img, bb = data['image'], data['annotation']
            img, bb = img.cuda(), bb.cuda()   
            pred = model(img)   
            loss = criterion(pred, bb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Loss for {} is {}".format(epoch, epoch_loss))
    print("Completed initial training. Unfreezing all layers")

    for param in model.parameters():
        param.requires_grad = True


    for epoch in range(epochs):
        epoch_loss = 0
        for data in dataloader:
            img, bb = data['image'], data['annotation']
            img, bb = img.cuda(), bb.cuda()
            pred = model(img)
            loss = criterion(pred, bb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Loss for {} is {}".format(epoch, epoch_loss))
 
