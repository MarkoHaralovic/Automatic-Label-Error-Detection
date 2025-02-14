import argparse
import labels
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from labels import Cityscapes
import logging
logging.basicConfig(level=logging.INFO)

def calc_polygon_area(polygon):
   """Shoelace Formula"""
   n = len(polygon)
   area = 0
   for i in range(n):
      x1, y1 = polygon[i]
      x2, y2 = polygon[(i + 1) % n]  # Wrap around to the first vertex
      area += (x1 * y2) - (x2 * y1)

   return abs(area) / 2


def probability_to_select(polygon_area):
   if not ( polygon_area >=500  and polygon_area <= 9500 ) :
      return -1
   return 1 * (10000 - np.abs(polygon_area)) / 9500


def get_permuted_polygons(polygonData, permutation_rate):
   assert permutation_rate >= 0 and permutation_rate <= 1

   prob_to_select = []
   for object in polygonData['objects']:
      polygon_area = calc_polygon_area(object['polygon'])
      prob_to_select.append(probability_to_select(polygon_area))

   num_to_select  = int(np.ceil(len(prob_to_select) * permutation_rate))
   prob_to_select = np.array(prob_to_select)
   label_ind_to_drop = np.argsort(prob_to_select)[::-1][:num_to_select]

   # prob_to_select = np.array(prob_to_select)
   # threshold = 1 - permutation_rate
   # label_ind_to_drop = np.where(prob_to_select > threshold)[0]

   polygonData_new = {
        'imgHeight': polygonData['imgHeight'],
        'imgWidth': polygonData['imgWidth'],
        'objects': [polygonData['objects'][i] for i in range(len(polygonData['objects'])) if i not in label_ind_to_drop]
    }


   return polygonData_new


def create_segmentation_mask(polygonData, name2label):
    """
    Creates a segmentation mask from a Cityscapes JSON annotation file.

    Args:
        polygonData (dict): Cityscapes json polygon annotation file.
        name2label (dict): Dictionary mapping class names to CityscapesClass objects.

    Returns:
        np.ndarray: Segmentation mask where each pixel is assigned a train_id.
    """
    height, width = polygonData["imgHeight"], polygonData["imgWidth"]

    segmentation_mask = np.full((height, width), 255, dtype=np.uint8)  # Default all to ignore

    mask_pil = Image.fromarray(segmentation_mask)
    draw = ImageDraw.Draw(mask_pil)

    missing_labels = set()

    for obj in polygonData["objects"]:
        if "deleted" in obj:
            continue

        label_name = obj["label"]

        if label_name not in name2label:
            if label_name.endswith("group"):
                label_name = label_name[:-len("group")]
            if label_name not in name2label:
                missing_labels.add(label_name)
                continue

        label = name2label[label_name]

        train_id = label.train_id

        poly_coord = [tuple(map(int, point)) for point in obj["polygon"]]

        draw.polygon(poly_coord, outline=train_id, fill=train_id)

    segmentation_mask = np.array(mask_pil, dtype=np.uint8)

    void_mask = (segmentation_mask == 255)

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for label in name2label.values():
        mask = segmentation_mask == label.train_id
        color_mask[mask] = label.color

    color_mask[void_mask] = (0, 0, 0)

    return color_mask


def create_preturbed_mask(json_file, name2label, permutation_rate):
   with open(json_file, 'r') as f:
      polygonData = json.load(f)
   polygonData = get_permuted_polygons(polygonData, permutation_rate)

   return create_segmentation_mask(polygonData, name2label)

def create_preturbed_dataset(json_folder, name2label, permutation_rate, save_dir):
   for json_file in tqdm(os.listdir(json_folder)):
      if json_file.endswith('.json'):
         mask = create_preturbed_mask(os.path.join(json_folder, json_file), name2label, permutation_rate)
         mask = Image.fromarray(mask)
         mask.save(os.path.join(save_dir, json_file.replace('.json', '_preturbed.png')))

def parse_args():
   parser = argparse.ArgumentParser(description='Permute labels in Cityscapes dataset')
   parser.add_argument('--json_folder', type=str, help='Path to the folder containing the json files')
   parser.add_argument('--permutation_rate', type=float, help='Permutation rate')
   parser.add_argument('--save_dir', type=str, help='Path to the folder to save the preturbed masks')

   args = parser.parse_args()
   return args

def main():
   args = parse_args()
   name2label = {label.name: label for label in Cityscapes.labels}
   create_preturbed_dataset(args.json_folder, name2label, args.permutation_rate, args.save_dir)

if __name__ == '__main__':
    main()