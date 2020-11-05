
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def dice_coef(y_pred, y_true):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coeff_multiclass(input,target,num_class):
    '''
    params
        input: predicted pixel classification, n*height*width
        target: ground truth, n*height*width
        num_class: num of class
    return
        dice for each class
    '''
    dices = np.zeros(num_class)
    
    for i in range(num_class):
        input_binary = np.array(input == i).astype(int)
        target_binary = np.array(target == i).astype(int)
        dice = dice_coef(input_binary,target_binary)
        dices[i] = dice

    return dices

def area_vector(pixel_label,num_class):
    '''
    params
        pixel_label: ground truth, n*height*width
        num_class: num of class
    return
        area for each class, n*num_class
    '''
    area_vector = np.zeros((pixel_label.shape[0],num_class))
    for i in range(num_class):
        pixel_label_binary = np.array(pixel_label == i).astype(int)
        pixel_nums = np.sum(pixel_label_binary,axis=(1,2))
        area_vector[:,i] = pixel_nums
    return area_vector

def mean_absolute_error(predict_vetor,label_vector):
    '''
        predict_vector: predict area vector, n*num_class
        label_vector: ground truth area vector, n*num_class
    '''
    n = predict_vetor.shape[0]
    return np.sum(np.absolute(predict_vetor - label_vector),0) / n

	
def aggregation_area(segment_area,estimate_area):
    '''
    combine segment area and estiamte area with uncertainty estiamtion
    params
        segment_area: batch_size*num_class
        estimate_area: batch_size*num_class
    '''
    u_segment_area = np.abs(segment_area-estimate_area)/np.abs(segment_area)
    u_estimate_area = np.abs(segment_area-estimate_area)/np.abs(estimate_area)
    weight_segment = 1 - u_segment_area / (u_segment_area + u_estimate_area)
    weight_estimate = 1 - u_estimate_area / (u_segment_area + u_estimate_area)
    
    return weight_segment * segment_area + weight_estimate * estimate_area

class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduce=True, size_average=True)

    def forward(self,predict_output,predict_area,label,device):
        '''
            predict_output: feature map of last layer, n*num_class*height*width
            predict_area: predicted areas, n*num_class
            label: segmentation ground truth, n*height*width
        '''
        
        batch_size,num_class,height,width = predict_output.size()
        pixel_num = torch.tensor(height*width).float()
        
        #-----calculate area of ground truth-----area_gt(n*num_class)
        area_gt = torch.zeros(batch_size,num_class).to(device)
        for i in range(num_class):
            area_gt[:,i] = (label == i).float().sum(axis=(1,2))
        area_gt = area_gt / pixel_num
        
        #-----calculate area by segmentation-----area_segment(n*num_Class)
        area_segment = torch.zeros(batch_size,num_class).to(device)
        predict_output_softmax = nn.functional.softmax(predict_output,dim=1) #n*num_Class*height*width
        predict_output_index = torch.max(predict_output_softmax, dim=1).indices #n*height*width

        for i in range(num_class):
            predict_right = (predict_output_index == i).float() * (label == i).float() # n*height*width     1-predict right  0-predict wrong
            area = (predict_right * predict_output_softmax[:,i,:,:]).sum(axis=(1,2)) #n*1
            area_segment[:,i] = area
        area_segment = area_segment / pixel_num

        #------normalize direct estimation of area---
        predict_area = predict_area / pixel_num


        #------calculate the loss-------------
        crossentropy_loss = self.crossentropy(predict_output,label)
        segment_gt_area_loss =  self.mse(area_segment,area_gt)
        estimate_gt_area_loss = self.mse(predict_area,area_gt)
        segment_estimate_area_loss = self.mse(area_segment,predict_area)

        return (crossentropy_loss + 0.05 * segment_gt_area_loss) + 0.5 * estimate_gt_area_loss + segment_estimate_area_loss
            

        



    