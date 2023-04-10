import os
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = "Cityscapes"      # Availabale in labels.py are Carla, Cityscape, Coco, PascalVOC
cfg.DATASET_DIR = "/home/marco/data/nvidia_val" # Root folder of your dataset

# Required input paths
cfg.NET_INPUT_DIR = os.path.join(cfg.DATASET_DIR, "net_input") # Required for visualizations
cfg.INFERENCE_OUTPUT_DIR = os.path.join(cfg.DATASET_DIR, "inference_output") # Segmentation masks generated by the net
cfg.GT_MASKS_DIR = os.path.join(cfg.DATASET_DIR, "gt_masks") # Ground truth segmentation masks
cfg.LOGITS_DIR = os.path.join(cfg.DATASET_DIR, "logits") # Ground truth segmentation masks
cfg.PERTURBED_MASKS_DIR = os.path.join(cfg.DATASET_DIR, "masks_perturbed")

# Paths to store intermediate results. Only the root has to be set
cfg.INTERMEDIATE_DIR = "/home/marco/Automatic-Label-Error-Detection/intermediate_results"
cfg.COMPONENTS_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "components")  
cfg.METRICS_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "metrics")  

# Visualization paths
cfg.VISUALIZATIONS_DIR = "/home/marco/Automatic-Label-Error-Detection/visualizations"
cfg.ERROR_PROPOSAL_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "error_proposals") # Where the proposals are saved

cfg.NUM_WORKERS = 8 # Number of multiprocessing workers
cfg.RANDOM_SEED = 1 # Control the Random split of the data
cfg.SPLIT_RATIO = 0.5 # Value between 0 and 1. Determine how much of the dataset is used to train meta Seg.
cfg.CLASS_IDS = [6, 7, 11, 12, 13, 14, 15, 17, 18] # Class ids of classes in which we search for label errors. E.g. Carla [2, 7, 9, 13] Cityscapes [6, 7, 11, 12, 13, 14, 15, 17, 18] #Pascal list(range(1, 21)) #Coco list(range(1, 93)) 
cfg.MIN_PROPOSAL_SIZE = 100 # Defines the minimum amount of pixel error a proposal must have
cfg.NUM_PRPOSALS = 100 # Maximum amount of error proposals

cfg.BENCHMARK = True
cfg.DIFF_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "diffs") # Folder for masks indicating differences between ground truth and perturbed ground truth
cfg.BENCHMARK_PROPOSAL_VIS_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "benchmark_proposals") # Where the proposals for the benchmark are saved
cfg.ERROR_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "label_errors") # Where the label error masks are saved
cfg.MIN_ERROR_SIZE = 250 # Defines the minimum amount of pixel error an label error must have. Only used for benchmarking
