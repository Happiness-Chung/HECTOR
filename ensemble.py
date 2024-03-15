import torch
import numpy as np
import random
import argparse
import itk
from itkwidgets import view

from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time
from model import *
from tqdm import tqdm
import torch.optim as optim
from random import sample 
from metrics import multiclass_dice_coeff, SegmentationMetrics
from datasets import Hector, HectorTest
import torch.nn.functional as F
import torchio as tio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from datetime import datetime

def dice_score(preds, labels, epsilon=1e-3, index=None):
    """
    Compute the Dice Score.
    
    Args:
        preds (Tensor): Predicted outputs from the model, with values in [0, 1].
        labels (Tensor): Ground truth binary labels, with values {0, 1}.
        epsilon (float): Small constant for numerical stability.

    Returns:
        float: Dice Score.
    """
    total_dice = 0
    total_precision = 0
    total_recall = 0
    preds = (preds > 0.5).float()

    # Convert the numpy array to an ITK image
    segmentation_image = itk.image_from_array(preds[0][0].cpu().numpy().astype(np.uint8))
    print("preds{}: ".format(index), preds[0][0].cpu().numpy().astype(np.uint8).sum())
    # Visualize the ITK image
    itk.imwrite(segmentation_image, "pred{}.mha".format(index))
    # Convert the numpy array to an ITK image
    segmentation_image = itk.image_from_array(labels[0][0].cpu().numpy().astype(np.uint8))
    print("Labels{}: ".format(index), labels[0][0].cpu().numpy().astype(np.uint8).sum())
    # Visualize the ITK image
    itk.imwrite(segmentation_image, "label{}.mha".format(index))

    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    for i in range(len(preds)):
        
    # Flatten label and prediction tensors
        preds_flat = preds[i].view(-1)
        labels_flat = labels[i].view(-1)
        
        # Calculate intersection and union
        true_positives = (preds_flat * labels_flat).sum()
        false_positives = (preds_flat * (1 - labels_flat)).sum()
        false_negatives = ((1 - preds_flat) * labels_flat).sum()

        union = preds_flat.sum() + labels_flat.sum()
        
        # Compute Dice score
        dice = (2. * true_positives + epsilon) / (union + epsilon)
        precision = (true_positives + epsilon) / (true_positives + false_positives + epsilon)
        recall = (true_positives + epsilon) / (true_positives + false_negatives + epsilon)

        total_dice += dice
        total_precision += precision
        total_recall += recall

    return (total_dice /len(preds)) , (total_precision / len(preds)), (total_recall / len(preds))

device = 'cuda:0'
class_num = 1
datasets = []
global best_validation_score
best_validation_score = 0

seed = 1999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = Hector()
indices_ct = list(range(224))
indices_pt = list(range(448,672))
np.random.shuffle(indices_ct)
np.random.shuffle(indices_pt)
test_indices = indices_ct[24:48] + indices_pt[24:48]

test_dataset = tio.SubjectsDataset([dataset[i] for i in test_indices])
test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True)
criterion = nn.BCELoss()

epoch_loss = []
sigmoid = nn.Sigmoid()
Model = UNet3D(1, class_num).to(device)
model_pathes = [
    # "History/2024-02-05_9H/best_validation_model.pth", # alone
    # "History/2024-02-07_8H/best_validation_model.pth", # alone
    # "History/2024-02-08_9H/best_validation_model.pth", # alone
    # "History/2024-02-09_14H/best_validation_model.pth", # alone
    # "History/2024-02-10_9H/best_validation_model.pth", # alone
    # "History/2024-02-13_8H/best_validation_model.pth", # NC
    # "History/2024-02-14_8H/best_validation_model.pth" # NC
    # "History/2024-02-18_7H/best_validation_model.pth",
    "History/2024-02-21_21H/best_test_model.pth",

]
metric = SegmentationMetrics(ignore_background= False)

Model.eval()
batch_loss = []
total_precision = 0
total_recall = 0
total_dice_score = 0
batch_loss = []
# wandb.watch(Model, log='all', log_freq=10)
batch_loss = []
total_precision = 0
total_recall = 0
total_dice_score = 0
batch_loss = []
# for batch_idx, data in enumerate(tqdm(test_data, desc = 'Processing: ')):
for batch_idx, data in enumerate(test_data):
    # logging.info(images.shape)
    masks_preds = []
    images = data['IMAGE']['data'].float()
    labels = data['LABEL']['data'].float()
    images, labels = images.to(device), labels.to(device)

    segmentation_image = itk.image_from_array(images[0][0].cpu().numpy().astype(np.uint8))
    # Visualize the ITK image
    itk.imwrite(segmentation_image, "image{}.mha".format(batch_idx))

    for i in range(len(model_pathes)):
        Model.load_state_dict(torch.load(model_pathes[i]))
        masks_preds.append(Model(images))
    masks_pred = torch.stack(masks_preds,dim=0)
    masks_pred = torch.mean(masks_pred, dim=0)
    # Model.load_state_dict(torch.load(model_pathes[i]))
    # masks_pred = Model(images)
    mask_true = labels.squeeze(1).type(torch.LongTensor)
    dice, precision, recall = dice_score(sigmoid(masks_pred).to(device), mask_true.type(torch.float).unsqueeze(1).to(device),index= batch_idx)

    total_recall += recall
    total_precision += precision
    total_dice_score += dice

total_recall /= len(test_data)
total_precision /= len(test_data)
total_dice_score /= len(test_data)

print("Test result: Dice Score(w/o b): {}, precision = {}, recall = {}*".format(total_dice_score, total_precision, total_recall))