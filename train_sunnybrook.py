import pydicom, cv2, re
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from unet import Unet
from helpers import center_crop, get_SAX_SERIES
from utils import dice_coeff_multiclass,MultiLoss,area_vector,mean_absolute_error
from dataloader import SunnybrookDataset



SUNNYBROOK_ROOT_PATH = '/home/hyt/bi-ventricle-segmentation/data/sunnybrook'

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')

VALID_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart2',
                            'ValidationDataContours')

VALID_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_validation')


if __name__== '__main__':

    
    device = torch.device('cuda:1')

    contour_type = 'i'
    crop_size = 128
    num_class = 2

    train_dataloader = DataLoader(dataset=SunnybrookDataset(TRAIN_IMG_PATH,TRAIN_CONTOUR_PATH,contour_type,crop_size),
                                    batch_size=2,
                                    shuffle=True,
                                    num_workers=4)
    validate_dataloader = DataLoader(dataset=SunnybrookDataset(VALID_IMG_PATH,VALID_CONTOUR_PATH,contour_type,crop_size),
                                    batch_size=2,
                                    shuffle=False,
                                    num_workers=4)

    unet = Unet(1,num_class,crop_size).to(device)
    
    criterion = MultiLoss()
    optimizer = optim.Adam(unet.parameters())

    epochs = 100

    for epoch in range(epochs):

        print("train epoch ", epoch+1)
        epoch_loss = 0

        for step,(img_batch,mask_batch) in enumerate(train_dataloader):
            
            img_batch = img_batch.float().to(device) # batch_size*channel*height*width
            mask_batch = mask_batch.long().to(device) # batch_size*height*width
            
            optimizer.zero_grad()
            
            #-------forward----------
            predict_map,predict_area = unet(img_batch) # batch_size*num_class*height*width, batch_size*num_class

            loss = criterion(predict_map, predict_area, mask_batch, device)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print("epoch %d , average train loss %f" % ( epoch+1 , epoch_loss/(step+1) ))

        #-----------save model parameter-----
        if (epoch+1)%10 == 0:
            torch.save(unet.state_dict(),os.path.join('train_log',str(epoch+1)+'.pt'))
        

        # ----------validation------------

        unet.eval()

        dice_sum = np.zeros(num_class)
        estimate_mae_sum = np.zeros(num_class)
        segment_mae_sum = np.zeros(num_class)

        for step,(img_valid,mask_valid) in enumerate(validate_dataloader):

            img_valid = img_valid.float().to(device) #batch_size*channel*height*width
            
            #---------predict--------
            predict_map_valid,predict_area_valid = unet(img_valid)

            #--------calculate dice coefficient
            predict_pxiel_class = torch.max(nn.functional.softmax(predict_map_valid,dim=1), dim=1).indices 
            predict_pxiel_class = predict_pxiel_class.cpu().numpy() # batch_size*height*width
            
            mask_valid = mask_valid.numpy() #batch_size*height*width

            dice = dice_coeff_multiclass(predict_pxiel_class,mask_valid,num_class)
            dice_sum += dice

            #--------calculate mean absolute error of direct area estimation
            predict_area_valid = predict_area_valid.cpu().detach().numpy() #batch_size*num_class
            gt_area_valid = area_vector(mask_valid, num_class) #batch_size*num_class
            estimate_mae = mean_absolute_error(predict_area_valid, gt_area_valid)
            estimate_mae_sum += estimate_mae

            #--------calculate mean absolute error of area by segmentation
            segment_area_valid = area_vector(predict_pxiel_class,num_class)
            segment_mae = mean_absolute_error(segment_area_valid, gt_area_valid)
            segment_mae_sum += segment_mae


 
        print("average validate dice ",dice_sum/(step+1), "average validate mae ", estimate_mae_sum/(step+1), segment_mae_sum/(step+1))


        





