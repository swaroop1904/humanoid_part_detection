import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import cv2

from data_generator import detection_dataset
from model import Model


def fit(model, optimizer, num_epochs)
    losses = []
    accuracies = []
    fdrs = []
    recalls = []

    criterion = nn.MSELoss()
    criterion.cuda()
    
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            img, annotation = data['image'], data['annotation']
            img, annotation = img.cuda(), annotation.cuda()   
            pred = model(img)   
            loss = criterion(pred, annotation)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        losses.append(epoch_loss)
        print("loss for epoch{} is {}".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            model.eval()
            tp_epoch, fp_epoch, fn_epoch = 0, 0, 0
            for data in val_dataloader:
                img, annotation = data['image'], data['annotation']
                img, annotation = img.cuda(), bb.cuda()
                pred = model(img)
                tp, fp, tn, fn = detection_metrics(pred, annotation)
                tp_epoch += tp
                fp_epoch += fp
                fn_epoch += fn
            acc = (tp_epoch) / (tp_epoch + fp_epoch + fn_epoch)
            recall = tp_epoch / (tp_epoch + fn_epoch)
            # confirm the formula: false discovery rate = fp / tp + fn
            fdr = fp_epoch / (fp_epoch + tp_epoch)
            accuracies.append(acc)
            recalls.append(recall)
            fdrs.append(fdr)
            print("Acc for epoch {} is {}".format(epoch, acc))
            print("FDR for epoch {} is {}".format(epoch, fdr))
            print("Recall for epoch {} is {}".format(epoch, recall))


def detection_metrics(out, annotation):
    '''
    Returns the true positive, false positive, true negative and false negative values
    ''' 
    tp, fp, fn = 0,0,0
    
    for idx, img_out in enumerate(out):
        detected_parts = get_part_centers(img_out)
        gt_parts = get_part_centers(annotation[idx])

        for detected_part, gt_part in zip(detected_parts, gt_parts):
            for val in detected_part:
                if not val: continue
                x = val[0]
                y = val[1]
                for gt_x, gt_y in gt_part:
                    if np.abs(x-gt_x) < 4 and np.abs(y-gt_y) < 4:
                        tp += 1
                        break
                else:
                    fp += 1
                    
            for gt_x, gt_y in gt_part:
                for x,y in detected_part:
                    if np.abs(x-gt_x) < 4 and np.abs(y-gt_y) < 4:
                        break
                else:
                    fn += 1
    return tp, fp, fn

def get_part_centers(img_out):
    '''
    Decodes the network output for one image
    '''
    out = img_out.numpy()
    out = np.uint8(out * 255)
    
    detected_parts = []
    for part in out:
        blurred_part = cv2.GaussianBlur(part, (5, 5), 0)
        thresh_part = cv2.threshold(blurred_part, 60, 255, cv2.THRESH_BINARY)[1]
        cnts_part = cv2.findContours(thresh_part, cv2.RETR_EXTERNAL,
               cv2.CHAIN_APPROX_SIMPLE)
        cnts_part = imutils.grab_contours(cnts_part)

        part_loc = []
        for c in cnts_part:
            M = cv2.moments(c)
            c_x = int(M["m10"] / M["m00"]) 
            c_y = int(M["m01"] / M["m00"]) 
            part_loc.append([c_x, c_y])
        detected_parts.append(part_loc)
    return detected_parts

if __name__ == '__main__':

    # train only the decoder weights for 50 epochs 
    csv_path = '/home/sbhand2s/downloads/vision_labs_project/files/dataset/Deepansh_Swaroop/labels.csv'
    img_path = '/home/sbhand2s/downloads/vision_labs_project/files/dataset/Deepansh_Swaroop/images/'

    resnet = resnet18(pretrained=True)
    resnet.avgpool = Identity()
    resnet.fc = Identity()

    for param in resnet.parameters():
        param.requires_grad = False

    model = Model(resnet)

    train_dg = detection_dataset((640,480), img_path, csv_path)
    train_dataloader = DataLoader(dg, batch_size=20, shuffle=True)

    val_dg = detection_dataset((640,480), img_path, csv_path)
    val_dataloader = DataLoader(dg, batch_size=20, shuffle=True)

    optimizer = Adam(model.parameters())
    model.cuda()

    fit(model, optimizer, epochs=50)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters())
    fit(model, optimizer, epochs=50)
