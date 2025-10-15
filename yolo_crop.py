from custom_yolo_predictor.custom_detection_predictor import CustomDetectionPredictor
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List
import argparse
import os

import piexif
import cv2
from PIL import Image

def crop_with_yolo(image: np.array, shape: List[int], coords: List[int], margin_of_error: int) -> np.array:
    """
    Crop image with YOLO coordinates xyxy

    Args: 
        image (np.array): image to crop from
        shape (List[int, int]): shape of the image
        coords: List[int, int, int, int]: coordinates
        margin_of_error (int): creates "padding" for the crop

    returns: 
        (np.array): cropped in image
    """
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    row, col = shape

    # Ensure the new coordinates stay within the image boundaries
    final_x1 = max(0,   x1 - margin_of_error)
    final_y1 = max(0,   y1 - margin_of_error)
    final_x2 = min(col, x2 + margin_of_error)
    final_y2 = min(row, y2 + margin_of_error)

    return image[int(final_y1):int(final_y2), int(final_x1):int(final_x2)], (int(final_x1), int(final_y1), int(final_x2), int(final_y2))

def create_dir(path: str):
    """ 
    Creates directory if it does note exists
    Args: 
        path: directory to create
    """ 
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_and_metadata(pil_image: Image, dest_path: str, x1: int, y1: int, x2: int, y2: int) -> None: 
    """
    Save PIL image with coordinates metadata (for image reconstruction during holistic model evaluation)
    Args: 
        pil_image (Image): image to save with metadata
        dest_path (str): destination path for the image to save 
        x1 (int): coordinates to save on meta data
        y1(int): coordinates to save on meta data
        x2 (int): coordinates to save on meta data
        y2(int): coordinates to save on meta data
    """
    exif = pil_image.getexif()
    exif_bytes = exif.tobytes()
    exif_dict = piexif.load(exif_bytes)

    # Convert the custom metadata to a format that can be written in EXIF
    new_data = {
        piexif.ExifIFD.UserComment: f"{x1},{y1},{x2},{y2}".encode('utf-8')
    }
    
    exif_dict["Exif"].update(new_data)
    
    # Create the bytes for writing to the image
    exif_bytes = piexif.dump(exif_dict)
    
    # Save the image with the new EXIF data
    pil_image.save(dest_path, exif=exif_bytes)

def crop_from_yolo(image_results: List, label_split_dir: str, image_dest_dir: str, label_dest_dir: str) -> None: 
    """
    Crop image using bounding boxes and create new 'yolo_cropped/' datasets
    
    Args: 
        image_results (List[Result]): list of YOLO Results objects
        label_split_dir (str):  current ["test", "train", "val"] label split directory
        image_split_dir (str):  current ["test", "train", "val"] image split directory
        image_dest_dir (str):   image destination directory
        label_dest_dir (str):   label destination directory
    
    """
    global TOTAL_PREDICTIONS
    for result in image_results: 
        boxes = result.boxes
        image_path = result.path

        # If there's a prediction... 
        if boxes: 
            coords = boxes.xyxy
            # If there are multiple boxes, take the max/min (inclusive)
            if len(coords) > 1: 
                x1, x2 = torch.min(coords[:, 0]).item(), torch.max(coords[:, 2]).item()
                y1, y2 = torch.min(coords[:, 1]).item(), torch.max(coords[:, 3]).item()
            ### If there's a single box, take the first
            else: 
                coord = coords[0]
                x1, x2 = int(coord[0]), int(coord[2])
                y1, y2 = int(coord[1]), int(coord[3])
                
            basename = os.path.basename(image_path)
            label_path = os.path.join(label_split_dir, basename) # label path of the original label (to copy)
            dest_image_path, dest_label_path = os.path.join(image_dest_dir, basename), os.path.join(label_dest_dir, basename)

            # Resize the image smaller, such that we can crop it
            orig_img = cv2.resize(result.orig_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            row, col, _ = orig_img.shape

            # Crop image and labels with a margin of error
            cropped_image, final_coords = crop_with_yolo(orig_img, (row, col), (x1, y1, x2, y2), MARGIN_OF_ERROR)
            cropped_label, final_coords = crop_with_yolo(
                cv2.resize(cv2.imread(label_path, cv2.IMREAD_UNCHANGED), (row, col), interpolation=cv2.INTER_AREA),
                (row, col),  (x1, y1, x2, y2), MARGIN_OF_ERROR)
            
            # Save the cropped image with metadata (to be reconstructed in evaluation)
            save_image_and_metadata(Image.fromarray(cropped_image), dest_image_path, final_coords[0], final_coords[1], final_coords[2], final_coords[3])
            # Save the cropped label
            cv2.imwrite(dest_label_path, cropped_label) 
            
            TOTAL_PREDICTIONS+=1
            print(f"SAVING: Prediction in... {image_path}")
            print(f"SAVING: Prediction in... {label_path}")
            
        ### No prediction...
        else: 
            print(f"SKIPPING: No Prediction in... {image_path}")
    
def yolo_crop_async(): 
    global TOTAL_PREDICTIONS
    image_dir,      label_dir =         os.path.join(IN_DIR, "images"),     os.path.join(IN_DIR, "masks")
    image_dest_dir, label_dest_dir =    os.path.join(OUT_DIR, "images"),    os.path.join(OUT_DIR, "masks")

    for split in ["test", "train", "val"]:
        image_split,        label_split =       os.path.join(image_dir, split),         os.path.join(label_dir, split) 
        image_dest_split,   label_dest_split =  os.path.join(image_dest_dir, split),    os.path.join(label_dest_dir, split)
    
        # Construct Destination Directories
        create_dir(image_dest_split), create_dir(label_dest_split)


        # Construct the full directories of images and labels
        image_list = sorted( os.listdir(image_split) ) 
        image_full_paths = [os.path.join(image_split, image) for image in image_list]

        args = dict(conf=CONFIDENCE, save=False, verbose=True, device="cuda", imgsz=IMAGE_SIZE, batch=BATCH_SIZE)  
        predictor = CustomDetectionPredictor(overrides=args)
        predictor.setup_model(MODEL_DIR)

        ### ------------------------------------------
        ### Single Batch
        # for image_path in image_full_paths:
        #     image_results = predictor(image_path)
        #     crop_from_yolo(image_results, label_split, image_dest_split, label_dest_split)

        ### ------------------------------------------
        ### Multi Batch
        batches = [image_full_paths[i:i + BATCH_SIZE] for i in range(0, len(image_full_paths), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
            futures = []
            for batch in batches: 
                batch_results = predictor(batch)
                for result in batch_results: 
                    # Submit tasks and store the future
                    future = executor.submit(crop_from_yolo, [result], label_split, image_dest_split, label_dest_split)
                    futures.append(future)
            # Wait for all futures to complete before exiting the with block
            for future in as_completed(futures): 
                try: 
                    future.result() # This will raise any exceptions that occurred in the thread
                except Exception as e: 
                    print(f"Error processing heatmap: {e}")

    print(f"\nThere were a total of {TOTAL_PREDICTIONS} predictions...")

if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Performs YOLO cropping on a preprocessed BraTS 2D
    dataset, to prepare them segmentation training

    Creates new directory named yolo_cropped, containing
    all of the YOLO cropped images
    """
    # ---------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument("--model_dir", type=str,help='YOLO model directory\t[None]')
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')

    parser.add_argument('--batch_size', type=int, help='batch size to for YOLO inference (speeds up processing)\t[32]')
    parser.add_argument('--image_size', type=int, help='confidence for binarizing the image\t[160]')
    parser.add_argument('--confidence', type=int, help='confidence for binarizing the image\t[0.5]')
    parser.add_argument('--margin_of_error', type=int, help='amount of pixels to pad the crops (all sides) as a margin of error\t[30]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')

    parser.add_argument('--filter', action='store_true', help='Enable YOLO Gating, discard images under the confidence score')

    args = parser.parse_args()

    # Assign Global Variables
    IN_DIR = args.in_dir or "stacked_segmentation"
    OUT_DIR = args.out_dir or f"{IN_DIR}_cropped"
    MODEL_DIR = args.model_dir or "yolo_checkpoint/weights/best.pt"
    DEVICE = args.device or "cuda"
    BATCH_SIZE = args.batch_size or 32
    IMAGE_SIZE = args.batch_size or 160
    CONFIDENCE = args.confidence or 0.7
    WORKERS = args.workers or 10
    FILTER = args.filter or False
    MARGIN_OF_ERROR = args.margin_of_error or 30

    TOTAL_PREDICTIONS = 0
    yolo_crop_async()
