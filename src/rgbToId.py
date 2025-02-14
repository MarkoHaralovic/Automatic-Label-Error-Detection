import labels
import os
from PIL import Image
import numpy as np
from tqdm import tqdm 
import shutil
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def convertIdToInf(id_image, trainId2Color):
    """
    Converts an ID segmentation mask back to an RGB segmentation mask.

    Args:
        id_image (PIL Image or NumPy array): The ID mask (grayscale).
        trainId2Color (dict): Mapping from train IDs to RGB colors.

    Returns:
        PIL Image: RGB segmentation mask.
    """
    id_image = np.array(id_image, dtype=np.uint8)

    rgb_image = np.zeros((id_image.shape[0], id_image.shape[1], 3), dtype=np.uint8)

    id_flat = id_image.flatten()

    for id, color in trainId2Color.items():
        mask = id_flat == id  
        rgb_image.reshape(-1, 3)[mask] = color 

    return rgb_image


def convertInfToId(image, trainColor2Id):
    """
    Converts an RGB segmentation mask back to an ID mask.

    Args:
        image (PIL Image or NumPy array): The RGB mask.
        trainColor2Id (dict): Mapping from RGB colors to train IDs.

    Returns:
        np.ndarray: The ID mask.
    """
    image = np.array(image, dtype=np.uint8)  # Convert to NumPy array
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]

    height, width = image.shape[:2]
    id_image = np.full((height, width), 255, dtype=np.uint8)  # Default to ignored regions (255)

    # Ensure dictionary keys are tuples (avoid mutable lists)
    color_to_id = {tuple(color): train_id for color, train_id in trainColor2Id.items()}

    # Convert each pixel to the corresponding ID
    for color, train_id in color_to_id.items():
        mask = np.all(image == np.array(color), axis=-1)  # Find pixels matching this color
        id_image[mask] = train_id  # Assign correct train_id

    return id_image

def remapGT(gt_img, id2trainId):
    """
    Remaps the ground truth image using the id2trainId mapping.

    Args:
        gt_img (PIL Image or NumPy array): The ground truth image.
        id2trainId (dict): Mapping from original IDs to train IDs.

    Returns:
        PIL Image: The remapped ground truth image.
    """
    gt_img = np.array(gt_img, dtype=np.uint8)  
    mapping_array = np.zeros(256, dtype=np.uint8) 
    for id, trainId in id2trainId.items():
        mapping_array[id] = trainId
    remapped_gt_img = mapping_array[gt_img]  
    return remapped_gt_img


Dataset = getattr(labels, "Cityscapes")

segmentation_map_dir = r""
segmentation_id_dir = r""
segmentation_check_dir =  r""

def process_image(args):
   image, conversion, color_id_mapping, input_dir, output_dir = args
   image_path = os.path.join(input_dir, image)
   gt_img = Image.open(image_path)
   id_img = Image.fromarray(conversion(gt_img, color_id_mapping))
   id_img.save(os.path.join(output_dir, image))


def parallel_process_images(function_to_perform, conversion, color_id_mapping, input_dir, output_dir):
    images = os.listdir(input_dir)
    images = [image for image in images if image.endswith(".png")]
    args = [(image, conversion, color_id_mapping, input_dir, output_dir) for image in images]
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(function_to_perform, args), total=len(images)))

        
if __name__ == "__main__":
    parallel_process_images(process_image, 
                           remapGT,
                           Dataset.id2trainId,
                           segmentation_map_dir,
                           segmentation_id_dir
                           )
   
    parallel_process_images(process_image, 
                           convertInfToId,
                           Dataset.trainColor2Id,
                           segmentation_map_dir,
                           segmentation_id_dir
                           )
   
    parallel_process_images(process_image, 
                            convertIdToInf,
                            Dataset.trainId2color,
                            segmentation_id_dir,
                            segmentation_map_dir
                              )
