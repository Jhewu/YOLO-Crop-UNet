# Internal
import os
import argparse

# External
import cv2
import torch 
from tqdm import tqdm

from monai.metrics import DiceMetric

def evaluate_ensemble(mask_dir: str, label_dir: str) -> None:
    """
    Calculate the mean Dice score (with monai) with the two given directories
    Args: 
        mask_dir (str): directory containing reconstructed masks
        label_dir (str): directory containing the ground truth masks
    """

    # Get label paths
    label_paths = sorted([os.path.join(label_dir, i) for i in os.listdir(label_dir)])

    # Declare monai metrics (correctly)
    metric = DiceMetric(
            include_background = False, # exclude background when reporting Dice (standard practice)
            reduction="mean_batch",     
            get_not_nans = False, 
            ignore_empty = False, 
            num_classes = None,         # infers from data (will be 1 channel)
            return_with_label = False
        )
         
    metric.reset() # (not needed, but best practice)
    for i, label_path in enumerate(tqdm(label_paths)):
        label_name = os.path.basename(label_path)
        mask_path = os.path.join(mask_dir, label_name)

        pred = None
        # Load prediction
        if os.path.exists(MASK_PATH):
            pred = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert to tensor and NORMALIZE to [0, 1] range
            pred = torch.from_numpy(pred).float() / 255.0
            
            # Add channel and batch dimension
            pred = pred.unsqueeze(0).unsqueeze(0)
            
        else:
            # Create empty mask if prediction doesn't exist
            pred = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE) # Use correct size

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and NORMALIZE to [0, 1] range
        label = torch.from_numpy(label).float() / 255.0
        
        # Add channel and batch dimension
        label = label.unsqueeze(0).unsqueeze(0)
        
        # Update metrics (sigmoid -> binarization -> metric)
        pred_binary = (pred > 0.5).float()
        label_binary = (label > 0.5).float() # technically, it's not needed as they are already binarized when saving
        metric(pred_binary, label_binary)

    mean_dice = metric.aggregate().item()
    print(f"\nThe mean dice score is {mean_dice}")

if __name__ == "__main__":
    # -------------------------------------------------------------
    des="""
    Evaluate YOLO-UNet "ensembled" predictions by calculating the mean 
    Dice Score from the reconstructed masks
    """
    # -------------------------------------------------------------
    # TODO: Add argparse

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--split", type=str, help='split to evaluate\t[test]')
    parser.add_argument("-i", "--image_size", type=int, help='image size for evaluation\t[160]')
    parser.add_argument("-m", "--masks_path", type=str, help='path of reconstructed masks\t[reconstructed_test/labels]')
    parser.add_argument("-l", "--labels_path", type=str, help='path of ground truth labels\t[3_fold_dataset/stacked_segmentation_0/test]')
    args = parser.parse_args()

    # Set defaults
    SPLIT = args.split or "test"
    IMAGE_SIZE = args.image_size or 160
    MASK_PATH = args.masks_path or f"reconstructed_{SPLIT}/labels"
    LABEL_PATH = args.labels_path or "3_fold_dataset/stacked_segmentation_0/"
    
    evaluate_ensemble(mask_dir=os.path.join(MASK_PATH, SPLIT), label_dir=os.path.join(LABEL_PATH, SPLIT))
