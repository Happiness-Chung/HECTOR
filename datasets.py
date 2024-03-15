'''
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
'''

import glob
from torch.utils.data import Dataset
import torchio as tio
from numpy.core.fromnumeric import mean
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation, 
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
import math

training_transform = tio.Compose([
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(axes=(0,)),
    tio.RandomAffine(),
    tio.ZNormalization(),
])

validation_transform = tio.Compose([
    ZNormalization(),
])



def Hector(path = "C:/Users/user/Desktop/data/HECTOR2021/", hospitals = ["CHGJ", "CHMR", "CHUM", "CHUP"]):
        
    folder_path = path
    imgs = []
    labels = []

    # for i in range(len(hospitals)):
    # shuffle 했을 때 잘되는지 궁금...
    imgs = glob.glob('C:/Users/user/Desktop/data/HECTOR2021/crops/(2,2,2)/Images/*')
    labels = glob.glob('C:/Users/user/Desktop/data/HECTOR2021/crops/(2,2,2)/Labels/*')
    # print(len(imgs), len(labels))

    subjects = []
    for i in range(len(imgs)):
        subject = tio.Subject(
            IMAGE = tio.ScalarImage(imgs[i]),
            LABEL = tio.LabelMap(labels[i]),
        )

        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects)

def get_inverse_effective_number(alpha, beta, freq): # beta is same for all classes
        son = freq / alpha # scaling factor
        if son == 0:
            son= 1
            son = math.pow(beta,son)
        En =  (1 - beta) / (1 - son)
        return En # the form of vector

def dataset_count(indices):

    labels = glob.glob('C:/Users/user/Desktop/data/HECTOR2021/crops/(2,2,2)/Labels/*')
    selected_elements = [labels[i] for i in indices]
    pos = 0
    neg = 0
    beta = 0.9999
    alpha = 1

    for i in range(len(selected_elements)):
        label = nib.load(selected_elements[i]).get_fdata()
        if label.sum() > 0:
            pos += 1
        else:
            neg += 1

    pos_eff = get_inverse_effective_number(alpha, beta, pos)
    neg_eff = get_inverse_effective_number(alpha, beta, neg)

    pos_weight = pos_eff / (pos_eff + neg_eff)
    neg_weight = neg_eff / (pos_eff + neg_eff)

    print("Pos weight: ", pos_weight, "Neg weight", neg_weight)
    print("Pos contribution: ", pos * pos_weight, "Neg contribution: ", neg * neg_weight)

    return pos_weight, neg_weight





def HectorTest(path = "C:/Users/user/Desktop/data/HECTOR2021/", hospital = "CHUS"):
        
    folder_path = path
    imgs = glob.glob(folder_path + 'npy_imgs/'+ hospital + '*')
    labels = glob.glob(folder_path + 'npy_labels/'+ hospital + '*')

    subjects = []
    for i in range(len(imgs)):
        subject = tio.Subject(
            MRI = tio.ScalarImage(imgs[i]),
            LABEL = tio.LabelMap(labels[i//2]),
        )

        subjects.append(subject)
    
    return tio.SubjectsDataset(subjects)

