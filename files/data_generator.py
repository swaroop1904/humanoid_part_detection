import torch
import numpy as np
from torch.utils.data import Dataset
import os
import csv
from collections import defaultdict
from skimage.io import imread

class detection_dataset(Dataset):
    def __init__(self, img_size, img_folder_path, csv_path, transform=True):
        self.height = img_size[1]
        self.width = img_size[0]
        self.folder_path = img_folder_path
        self.img_files = os.listdir(img_folder_path)
        self.csv_path = csv_path
        self.transform = transform
        self.get_annotations()
        
    def __len__(self):
        return (len(self.img_files))
    
    def __getitem__(self, index):
        img_name = self.img_files[index]
        img = imread(self.folder_path + img_name)
        annotation = self.annotations[img_name]
        annotation = annotation[:,::4,::4]
        if self.transform:
            img = img.transpose((2,0,1))
            img, annotation = torch.from_numpy(img), torch.from_numpy(annotation)
            img = img.type(torch.FloatTensor)
            annotation = annotation.type(torch.FloatTensor)
        sample = {'image':img, 'annotation':annotation}

        return sample
          
    def get_annotations(self):
        self.annotations = defaultdict(lambda: np.zeros([4, self.height, self.width]))
        part_mapping = {'Head':0, 'Foot':1, 'Hand':2, 'Trunk':3}
        with open(self.csv_path) as csvfile:
            readCSV = csv.reader(csvfile)
            for row in readCSV:
                img_name = row[0]
                image_annotation = self.annotations[img_name]
                channel = part_mapping[row[1].capitalize()]
                x = int(row[3])
                y = int(row[2])
                
                if image_annotation[channel, x-8:x+8, y-8:y+8].shape != (16,16): continue
                image_annotation[channel, x-8:x+8, y-8:y+8] = makeGaussian(16,8)
                

def makeGaussian(size, fwhm = 3):
    ''' 
    Input: size: length of a side of the square
           fwhm: full-width-half-maximum (effective radius)
    Output: a square gaussian kernel
    Reference: https://stackoverflow.com/a/14525830
    '''

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    x0 = y0 = size // 2

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
