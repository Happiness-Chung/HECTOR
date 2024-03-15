import pydicom
import numpy as np
import matplotlib.pyplot as plt

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    windowed_img = (windowed_img - img_min) / (img_max - img_min)
    return windowed_img

def visualize_dicom(dicom_file):
    # Read the DICOM file
    ds = pydicom.dcmread(dicom_file)
    
    # Extract pixel data
    pixel_array = ds.pixel_array
    windowed_image = window_image(pixel_array, 40, 400)
    
    # Plot the image
    plt.imshow(windowed_image, cmap=plt.cm.gray)
    plt.title('DICOM Image')
    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
dicom_file = 'C:/Users/user/Desktop/data/HaN/manifest-VpKfQUDr2642018792281691204/Head-Neck-PET-CT\HN-CHUM-003/08-27-1885-NA-TomoTherapy Patient Disease-80724/525144918.000000-kVCT Image Set-34203/1-001.dcm'
visualize_dicom(dicom_file)