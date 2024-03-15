import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.transform import resize
import matplotlib.cm as cm

path = 'C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Images/*'
path_label = 'C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Labels/*'
save_path = 'C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Images/'
save_path_label = 'C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Labels/'

imgs_path = glob.glob(path)
labels_path = glob.glob(path_label)
ct_cnt = 0
pt_cnt = 0

for i in range(224):

    # index = int(imgs_path[i].split('/')[-1].split('.')[0].split('\\')[-1])
    
    # if index % 2 == 0:
    img = nib.load(imgs_path[i]).get_fdata()
    clipped_ct = np.clip(img, -100 , 300)
    # normalized_ct = (clipped_ct - clipped_ct.mean()) / clipped_ct.std()
    # print(normalized_ct.min(), normalized_ct.max())
    nii_ct = nib.Nifti1Image(clipped_ct, affine = np.eye(4))
    nib.save(nii_ct, save_path + "{}.nii.gz".format(ct_cnt))
    
    # for j in range(len(labels_path)):
    #     index_label = int(labels_path[j].split('/')[-1].split('.')[0].split('\\')[-1]) * 2
    #     if index_label == index:
    #         label = nib.load(labels_path[j])
    #         npy_label = nib.load(labels_path[j]).get_fdata()
    #         print("sum: ", npy_label.sum())
    #         nib.save(label, save_path_label + "ct_{}.nii.gz".format(ct_cnt))

                # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
                # ax1.imshow(normalized_ct[:, :, normalized_ct.shape[1]//2], cmap = cm.hot)
                # ax1.set_title('Original CT')
                # ax2.imshow( npy_label[:, :, npy_label.shape[1]//2], cmap = cm.hot)
                # ax2.set_title('label CT')
                # ax3.imshow( normalized_ct[:, :, normalized_ct.shape[1]//2] * 0.5 + 0.5 * npy_label[:, :, npy_label.shape[1]//2], cmap = cm.hot)
                # ax3.set_title('label PET')
                # plt.show()
    # ct_cnt += 1

    # if index % 2 == 1:
    #     img = nib.load(imgs_path[i])
    #     # clipped_ct = np.clip(img, -100 , 300)
    #     # normalized_ct = (img - img.mean()) / img.std()
    #     # nii_ct = nib.Nifti1Image(normalized_ct, affine = np.eye(4))
    #     nib.save(img, save_path + "pt_{}.nii.gz".format(pt_cnt))
        
    #     for j in range(len(labels_path)):
    #         index_label = int(labels_path[j].split('/')[-1].split('.')[0].split('\\')[-1]) * 2 + 1
    #         if index_label == index:
    #             label = nib.load(labels_path[j])
    #             npy_label = nib.load(labels_path[j]).get_fdata()
    #             # print(npy_label.sum())
    #             nib.save(label, save_path_label + "pt_{}.nii.gz".format(pt_cnt))

                # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
                # ax1.imshow(img[:, :, img.shape[1]//2], cmap = cm.hot)
                # ax1.set_title('Original PET')
                # ax2.imshow(normalized_ct[:, :, normalized_ct.shape[1]//2], cmap = cm.hot)
                # ax2.set_title('Standardized PET')
                # ax3.imshow( npy_label[:, :, npy_label.shape[1]//2], cmap = cm.hot)
                # ax3.set_title('label PET')
                # plt.show()
        
        # pt_cnt += 1
        
        # ax3.imshow(normalized_ct[:, :, normalized_ct.shape[1]//2], cmap = cm.hot)
        # ax3.set_title('Standardized CT')
        # fig.savefig('Preprocessing{}'.format(index))
    # print(clipped_ct.min(), clipped_ct.max())
    # if clipped_ct.min() == 0 and clipped_ct.max() == 0:
    #     os.remove(imgs_path[i])
    #     cnt += 1
  

#print(cnt)
