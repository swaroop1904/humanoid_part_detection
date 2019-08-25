import torch
from torchvision.models import resnet18
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet = resnet18(pretrained=True)
        resnet.avgpool = Identity()
        resnet.fc = Identity()

        self.downsampling = resnet

        self.cat1 = nn.Conv2d(64, 128, 1)
        self.cat2 = nn.Conv2d(128, 128, 1)
        self.cat3 = nn.Conv2d(256, 128, 1)
        
        self.upsampling_1 = nn.Sequential(nn.ReLU(),
                                          nn.ConvTranspose2d(512, 128, 3, 2, 1, 1)
                                         )

        self.upsampling_2 = nn.Sequential(nn.ReLU(),
                                          nn.BatchNorm2d(256),
                                          nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
                                         )

        self.upsampling_3 = nn.Sequential(nn.ReLU(),
                                          nn.BatchNorm2d(256),
                                          nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
                                         )

        self.upsampling_4 = nn.Sequential(nn.ReLU(),
                                         nn.BatchNorm2d(256),
                                         nn.ConvTranspose2d(256, 4, 3, 1, 1) 
                                        )

    def forward(self, x):
        x = self.downsampling.conv1(x)
        x = self.downsampling.bn1(x)
        x = self.downsampling.relu(x)
        x = self.downsampling.maxpool(x)

        #print("before block layers", x.shape)
        down1 = self.downsampling.layer1(x)
        #print("after 1st downsampling", down1.shape)
        cat1 = self.cat1(down1)
        down2 = self.downsampling.layer2(down1)
        #print("after second downsampling", down2.shape)
        cat2 = self.cat2(down2)
        down3 = self.downsampling.layer3(down2)
        #print("after third downsampling", down3.shape)
        cat3 = self.cat3(down3)
        down4 = self.downsampling.layer4(down3)  

        #print("after final downsampling", down4.shape)
        up1 = self.upsampling_1(down4)
        #print("after 1st upsampling", up1.shape)
        up2 = self.upsampling_2(torch.cat((up1, cat3), 1))
        #print("after second upsampling", up2.shape)
        up3 = self.upsampling_3(torch.cat((up2, cat2), 1))
        #print("after third upsampling", up3.shape)
        up4 = self.upsampling_4(torch.cat((up3, cat1), 1))
        #print("final upsampling", up4.shape)
        return up4       
