# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:05:10 2018

@author: Shubham
"""

import os
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import image2Tensor, tensor2Image, get_params

# class containing the fast gradient sign 
# untargeted attack 
class FastGradientSignTargeted:
    
    def __init__(self, model, eps):
        self.model = model
        self.eps = eps
        if not os.path.exists('../generated'):
            os.makedirs('../generated')
        
    def generate(self, original_image, org_class,
                 target_class):
        im_label = Variable(torch.from_numpy(np.asarray([target_class])))
        
        # define loss functions
        ce_loss = nn.CrossEntropyLoss()
        
        # process image
        processed_image = preprocess_image(original_image)
        
        # iteration for updates
        for i in range(10):
            print ('Iteration : %d' %i)
            processed_image.grad = None
            out = self.model(processed_image)
            pred_loss = ce_loss(out, im_label)
            pred_loss.backward()
            adv_noise = self.eps * torch.sign(processed_image.grad.data)
            processed_image.data = processed_image.data - adv_noise
            recreated_image = recreate_image(processed_image)
            prep_confirmation_image = preprocess_image(recreated_image)
            new_label = self.model(prep_confirmation_image)
            _, confirmation_prediction = confirmation_out.data.max(1)
            
            confirmation_confidence = nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            conf_pred = confirmation_prediction.numpy()[0]
            
            if conf_pred == target_class:
                noise_label = original_image - recreated_image
                cv2.imwrite('../generated/targeted_adv_noise_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('../generated/targeted_adv_img_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', recreated_image)
                break
        return 1
    
# Untargeted attack till the class label is not correctly identified
class FastGradientSignUntargeted:
    
    def __init__(self, model, eps):
        self.model = model
        self.eps = eps
        if not os.path.exists('../generated'):
            os.makedirs('../generated')
        
    def generate(self, original_image, org_class,
                 target_class):
        im_label = Variable(torch.from_numpy(np.asarray([target_class])))
        
        # define loss functions
        ce_loss = nn.CrossEntropyLoss()
        
        # process image
        processed_image = preprocess_image(original_image)
        
        # iteration for updates
        for i in range(10):
            print ('Iteration : %d' %i)
            processed_image.grad = None
            out = self.model(processed_image)
            pred_loss = ce_loss(out, im_label)
            pred_loss.backward()
            adv_noise = self.eps * torch.sign(processed_image.grad.data)
            processed_image.data = processed_image.data - adv_noise
            recreated_image = recreate_image(processed_image)
            prep_confirmation_image = preprocess_image(recreated_image)
            new_label = self.model(prep_confirmation_image)
            _, confirmation_prediction = confirmation_out.data.max(1)
            
            confirmation_confidence = nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            conf_pred = confirmation_prediction.numpy()[0]
            
            if conf_pred != im_label:
                noise_label = original_image - recreated_image
                cv2.imwrite('../generated/targeted_adv_noise_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('../generated/targeted_adv_img_from_' + str(org_class) + '_to_' +
                            str(confirmation_prediction) + '.jpg', recreated_image)
                break
        return 1    
    
if __name__ == "__main__":
    target_example = 0
    (original_image, prep_img, org_class, _, pretrained_model) =\
        get_params(target_example)
    target_class = 62  # Mud turtle
    
    FGS_untargeted = FastGradientSignTargeted(pretrained_model, 0.01)
    FGS_untargeted.generate(original_image, org_class, target_class)