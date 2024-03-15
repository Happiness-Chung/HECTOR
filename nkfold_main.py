'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import torch
import numpy as np
import random
import argparse

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
from datasets import Hector, HectorTest, dataset_count
import torch.nn.functional as F
import torchio as tio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from datetime import datetime

import wandb

parser = argparse.ArgumentParser(description='HECTOR Challenge Training')
parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='c07987db95186aade1f1dd62754c86b4b6db5af6', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='hb0522', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='Focal + Dice Loss', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='HECTOR', type=str, help='your wandb project name (you have to change)')
args = parser.parse_args()

device = 'cuda:0'
class_num = 1
Epochs = 50
kfold= 4
datasets = []
global best_validation_score
global best_test_score
best_validation_score = 0
best_test_score = 0


class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: Balance between positive and negative classes.
        :param gamma: Modulating factor to focus on hard examples.
        :param reduction: Method for reducing loss to a scalar ('none', 'mean', 'sum').
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Predicted probabilities (after sigmoid) of shape (N, *)
        :param targets: Ground truth binary masks of shape (N, *)
        """
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def test():

    Model.eval()
    batch_loss = []
    total_precision = 0
    total_recall = 0
    total_dice_score = 0
    batch_loss = []
    global best_test_score
    # wandb.watch(Model, log='all', log_freq=10)

    for batch_idx, data in enumerate(tqdm(test_data, desc = 'Processing: ')):
        # logging.info(images.shape)
        images = data['IMAGE']['data'].float()
        labels = data['LABEL']['data'].float()
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        masks_pred = Model(images)
        mask_true = labels.squeeze(1).type(torch.LongTensor)
        dice, precision, recall = dice_score(sigmoid(masks_pred).to(device), mask_true.type(torch.float).unsqueeze(1).to(device))

        total_recall += recall
        total_precision += precision
        total_dice_score += dice
    
    if total_dice_score > best_test_score:
        best_test_score = total_dice_score
        torch.save(Model.state_dict(), result_dir + "/best_test_model.pth")

    total_recall /= len(test_data)
    total_precision /= len(test_data)
    total_dice_score /= len(test_data)

    print("Test result: Dice Score(w/o b): {}, precision = {}, recall = {}*".format(total_dice_score, total_precision, total_recall))

def validation():

    Model.eval()
    total_precision = 0
    total_recall = 0
    total_dice_score = 0
    batch_loss = []
    # wandb.watch(Model, log='all', log_freq=10)
    global best_validation_score
    
    for batch_idx, data in enumerate(tqdm(validation_data, desc = 'Processing: ')):
        # logging.info(images.shape)
        images = data['IMAGE']['data'].float()
        labels = data['LABEL']['data'].float()
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        masks_pred = Model(images)
        mask_true = labels.squeeze(1).type(torch.LongTensor)
        dice, precision, recall = dice_score(sigmoid(masks_pred).to(device), mask_true.type(torch.float).unsqueeze(1).to(device))
        loss1 = criterion(sigmoid(masks_pred).to(device), mask_true.type(torch.float).unsqueeze(1).to(device))
        loss2 = dice_loss(sigmoid(masks_pred).to(device), mask_true.type(torch.float).unsqueeze(1).to(device))
        loss = loss1 + loss2

        total_recall += recall
        total_precision += precision
        total_dice_score += dice

        batch_loss.append(loss.item())

    total_recall /= len(validation_data)
    total_precision /= len(validation_data)
    total_dice_score /= len(validation_data)

    if total_dice_score > best_validation_score:
        best_validation_score = total_dice_score
        torch.save(Model.state_dict(), result_dir + "/best_validation_model.pth")

    print("Validation result: Dice Score(w/o b): {}, precision = {}, recall = {}*".format(total_dice_score, total_precision, total_recall))

    f = open(result_dir + "/validation_performance.txt", "a")
    f.write(str(sum(batch_loss) / len(batch_loss)) + ', ' + str(total_dice_score.cpu().item()) + ', ' + str(total_precision.cpu().item()) + ', ' + str(total_recall.cpu().item()) + "\n")
    f.close()   
    # wandb.log(
    #     {"Validation Loss": sum(batch_loss) / len(batch_loss)}
    # )

def dice_loss(prediction, target, epsilon = 1e-3):
    
    intersection =  (prediction * target).sum()
    whole = prediction.sum() + target.sum()
    soft_dice_loss = (2 * intersection + epsilon) / (whole + epsilon)
    
    return 1 - soft_dice_loss

def pnb_dice_loss(prediction, target, pos_weight, neg_weight, mu, epsilon = 1e-3): # PNB version
    
    total = 0
    for i in range(len(prediction)):
        intersection =  (prediction[i] * target[i]).sum()
        whole = prediction.sum() + target.sum()
        soft_dice_loss = (2 * intersection + epsilon) / (whole + epsilon)
        if target.sum() > 0:
            total += soft_dice_loss * pos_weight
        else:
            total += soft_dice_loss * neg_weight
    
    loss = mu * (1 - total / len(prediction))
    
    return loss

def dice_score(preds, labels, epsilon=1e-3):
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


seed = 1999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dataset = Hector()
indices_ct = list(range(224))
indices_mix = list(range(225,448))
indices_pt = list(range(448,672))
np.random.shuffle(indices_ct)
np.random.shuffle(indices_mix)
np.random.shuffle(indices_pt)

epoch_loss = []
sigmoid = nn.Sigmoid()
Model = UNet3D(1, class_num).to(device)
optimizer = torch.optim.Adam(Model.parameters(), lr = 0.001 ,weight_decay=0.0001)
criterion = BinaryFocalLoss().to(device)
metric = SegmentationMetrics(ignore_background= False)
# scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,total_iters=200, power=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=0.00001)

# wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)
now = datetime.now()
result_dir = os.path.join('History', "{}_{}H".format(now.date(), str(now.hour)))
os.makedirs(result_dir, exist_ok=True)
# c = open(result_dir + "/config.txt", "w")
# c.write("plus: {}, depth: {}, dataset: {}, epochs: {}, lr: {}, momentum: {},  weight-decay: {}, seed: {}".format(args.plus, str(args.depth), args.dataset, str(args.epochs), str(args.lr), str(args.momentum),str(args.weight_decay), str(args.seed)))
open(result_dir + "/training_performance.txt", "w")
open(result_dir + "/validation_performance.txt", "w")

chunk = [50,100,150,200,224]
mu = 2

for k in range(kfold):

    train_indices = indices_ct[:24] + indices_ct[48:chunk[k]] + indices_ct[chunk[k+1]:] + indices_mix + indices_pt[48:chunk[k]] + indices_pt[:24] + indices_pt[chunk[k+1]:]
    # train_indices = indices_ct[:24] + indices_ct[48:] + indices_mix + indices_pt[48:] + indices_pt[:24] 
    validation_indices = indices_ct[chunk[k]:chunk[k+1]] + indices_pt[chunk[k]:chunk[k+1]]
    # validation_indices = indices_ct[:24] + indices_pt[:24]
    test_indices = indices_ct[24:48] + indices_pt[24:48]

    # CT
    # train_indices = indices_ct[:24] + indices_ct[48:chunk[k]] + indices_ct[chunk[k+1]:]
    # validation_indices = indices_ct[chunk[k]:chunk[k+1]] + indices_pt[chunk[k]:chunk[k+1]]
    # test_indices = indices_ct[24:48] + indices_pt[24:48]

    # PT
    # train_indices = indices_pt[48:chunk[k]] + indices_pt[:24] + indices_pt[chunk[k+1]:]
    # validation_indices = indices_ct[chunk[k]:chunk[k+1]] + indices_pt[chunk[k]:chunk[k+1]]
    # test_indices = indices_ct[24:48] + indices_pt[24:48]

    # test_dataset = HectorTest()
    train_dataset = tio.SubjectsDataset([dataset[i] for i in train_indices])
    validation_dataset = tio.SubjectsDataset([dataset[i] for i in validation_indices])
    test_dataset = tio.SubjectsDataset([dataset[i] for i in test_indices])
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 2, shuffle = True)
    validation_data = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True)
    # test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
    pos_weight, neg_weight = dataset_count(train_indices)
    criterion = nn.BCELoss()

    for epoch in range(Epochs):

        batch_loss = []
        total_precision = 0
        total_recall = 0
        total_dice_score = 0
        batch_loss = []
        # wandb.watch(Model, log='all', log_freq=10)
        Model.train()

        for batch_idx, data in enumerate(tqdm(train_data, desc = 'Processing: ')):
            # logging.info(images.shape)
            images = data['IMAGE']['data'].float()
            labels = data['LABEL']['data'].float()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()


            masks_pred = Model(images)
            mask_true = labels.type(torch.FloatTensor)
            mask_pred = (sigmoid(masks_pred) > 0.5).type(torch.float).squeeze()
            dice, precision, recall = dice_score(sigmoid(masks_pred).to(device), mask_true.to(device))
            loss1 = criterion(sigmoid(masks_pred).to(device), mask_true.to(device))
            loss = loss1 + dice_loss(sigmoid(masks_pred).to(device), mask_true.to(device))
            
            total_recall += recall
            total_precision += precision
            total_dice_score += dice
            
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        total_recall /= len(train_data)
        total_precision /= len(train_data)
        total_dice_score /= len(train_data)

        scheduler.step()

        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('---Local Training Epoch: {} \tLoss: {:.6f}'.format(epoch+1, sum(batch_loss) / len(batch_loss)))
            print("Training result: Dice Score: {}, precision = {}, recall = {}*".format( total_dice_score, total_precision, total_recall))
            # wandb.log({
            #     "Training Loss (dice loss)": sum(batch_loss) / len(batch_loss)
            # })
        f = open(result_dir + "/training_performance.txt", "a")
        f.write(str(sum(batch_loss) / len(batch_loss)) + ', ' + str(total_dice_score.cpu().item()) + ', ' + str(total_precision.cpu().item()) + ', ' + str(total_recall.cpu().item()) + "\n")
        f.close()   

        validation()

        test()