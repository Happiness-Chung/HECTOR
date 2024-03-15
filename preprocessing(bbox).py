from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk


path_input_images = "C:/Users/user/Desktop/data/HECTOR2021/hecktor_nii/"
path_input_labels = "C:/Users/user/Desktop/data/HECTOR2021/hecktor_nii/"
path_output_images = "C:/Users/user/Desktop/data/HECTOR2021/(1,1,1)/imgs_resampled_oropharynx/"
path_output_labels = "C:/Users/user/Desktop/data/HECTOR2021/(1,1,1)/labels_resampled_oropharynx/"

bbox_coordinates = pd.read_csv("C:/Users/user/Desktop/data/HECTOR2021/hecktor2021_bbox_training.csv")
# print(np.array(bbox_coordinates.loc[[0], ['x1', 'y1', 'z1']].values.tolist())[0])
# start = np.array(bbox_coordinates.loc[[0], ['x1', 'y1', 'z1']].values.tolist()[0])
# end = np.array(bbox_coordinates.loc[[0], ['x2', 'y2', 'z2']].values.tolist()[0])

@click.command()
@click.argument('input_image_folder',
                type=click.Path(exists=True),
                default=path_input_images)
@click.argument('input_label_folder',
                type=click.Path(exists=True),
                default=path_input_labels)
@click.argument('output_image_folder', default=path_output_images)
@click.argument('output_label_folder', default=path_output_labels)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1),
              help='Expect 3 positive floats describing the output '
              'resolution of the resampling. To avoid resampling '
              'on one or more dimension a value of -1 can be fed '
              'e.g. --resampling 1.0 1.0 -1 will resample the x '
              'and y axis at 1 mm/px and left the z axis untouched.')

def main(input_image_folder, input_label_folder, output_image_folder,
         output_label_folder, cores, resampling):
    """ This command line interface allows to resample NIFTI files within 
        the maximal bounding box covered by the field of view of both modalites 
        (PT and CT). The images are
        resampled with spline interpolation
        of degree 3 and the segmentation are resampled
        by nearest neighbor interpolation.
        INPUT_IMAGE_FOLDER is the path of the folder containing PT and CT images.
        INPUT_LABEL_FOLDER is the path of the folder containing the labels.
        OUTPUT_IMAGE_FOLDER is the path of the folder where to store the resampled PT and CT images.
        OUTPUT_LABEL_FOLDER is the path of the folder where to store the resampled labels.
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    input_image_folder = Path(input_image_folder).resolve() # 해당 파일이 존재하는 full directory 경로 반환
    input_label_folder = Path(input_label_folder).resolve()
    output_image_folder = Path(output_image_folder).resolve()
    output_label_folder = Path(output_label_folder).resolve()

    output_image_folder.mkdir(exist_ok=True)
    output_label_folder.mkdir(exist_ok=True)
    print('resampling is {}'.format(str(resampling)))

    patient_list = [
        f.name.split("_")[0] for f in input_image_folder.rglob("*_CT*") # 병원이름이랑인덱스 합친거 반환
    ]
    # print(patient_list)
    if len(patient_list) == 0:
        raise ValueError("No patient found in the input folder")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1]) # 출력 이미지의 방향을 변경하지 않고 그대로 출력
    resampler.SetOutputSpacing(resampling) # 출력이미지의 해상도를 조절하는 역할.

    def resample_one_patient(index, p):
        ct = sitk.ReadImage(
            str([f for f in input_image_folder.rglob(p + "_CT*")][0]))
        pt = sitk.ReadImage(
            str([f for f in input_image_folder.rglob(p + "_PT*")][0]))
        labels = [(sitk.ReadImage(str(f)), f.name)
                  for f in input_label_folder.glob(p + "*")]
        # bb = get_bouding_boxes(ct, pt)

        start = np.array(bbox_coordinates.loc[[index], ['x1', 'y1', 'z1']].values.tolist()[0])
        end = np.array(bbox_coordinates.loc[[index], ['x2', 'y2', 'z2']].values.tolist()[0])
        size = ((end - start) / resampling).astype(int)
        #size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        resampler.SetOutputOrigin(start)
        #resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        sitk.WriteImage(ct, str((output_image_folder / (p + "_CT.nii.gz"))))
        sitk.WriteImage(pt, str((output_image_folder / (p + "_PT.nii.gz"))))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        for label, name in labels:
            label = resampler.Execute(label)
            sitk.WriteImage(label, str((output_label_folder / name)))

    for i in range(len(patient_list)):
        resample_one_patient(i, patient_list[i])
    # with Pool(cores) as p:
    #    p.map(resample_one_patient, patient_list)


def get_bouding_boxes(ct, pt):
    """
    Get the bounding boxes of the CT and PT images.
    This works since all images have the same direction
    """

    ct_origin = np.array(ct.GetOrigin()) # 데이터의 원점 좌표를 튜플로 반환
    pt_origin = np.array(pt.GetOrigin())

    # print("origin: ", ct_origin)
    # print("origin: ", pt_origin)
    # print("size: ", ct.GetSize())
    # print("size: ", pt.GetSize())
    # print("spacing: ", ct.GetSpacing())

    ct_position_max = ct_origin + np.array(ct.GetSize()) * np.array(ct.GetSpacing()) #### Physical coordinates # The ogigin is the middle of the image.
    
    # print("max: ", ct_position_max)
    pt_position_max = pt_origin + np.array(pt.GetSize()) * np.array(pt.GetSpacing())
    # print("max: ", pt_position_max)


    return np.concatenate(
        [
            np.maximum(ct_origin, pt_origin),
            np.minimum(ct_position_max, pt_position_max),
        ],
        axis=0,
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()