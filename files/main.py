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
    csv_path = 'dataset/Deepansh_Swaroop/labels.csv'
    img_path = 'dataset/Deepansh_Swaroop/images/'

    dg = detection_dataset((640,480), img_path, csv_path)
    dataloader = DataLoader(dg, batch_size=20, shuffle=True)

    criterion = nn.MSELoss()
    model = Model()

    optimizer = Adam(model.parameters())

    model.cuda()
    criterion.cuda()

    epochs = 50

    for epoch in range(epochs):
        for data in dataloader:
            img, bb = data['image'], data['annotation']
            img, bb = img.cuda(), bb.cuda()   
            pred = model(img)   
            loss = criterion(pred, bb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
