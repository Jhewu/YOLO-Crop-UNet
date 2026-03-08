# Internal
import os
import argparse

# External
import cv2
import yaml 
import torch
import piexif
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Local
from unet import UNet
from dataset import SegmentationDataLoader

def create_dir(folder_name: str) -> None:
    """
    Creates given directory if it does not exist
    Args: 
        folder_name (str): directory to create
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

def read_metadata(img: Image):
    """
    Read the metadata from the YOLO cropped images
    Args: 
        img (Image): directory of image to read metadata from
    Returns: 
        List[int]: YOLO xyxy coordinates
    """
    exif = img.getexif()
    exif_bytes = exif.tobytes()
    exif_dict= piexif.load(exif_bytes)

    # Grab the raw bytes of the UserComment tag
    raw_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

    if raw_comment is None:
        return None

    # Convert the tuple (or bytes) to a real string
    # The EXIF spec says the first 8 bytes are an encoding prefix.
    # If you wrote the string yourself (without a prefix) it will
    # simply be the raw UTF‑8 bytes, so we can decode directly.
    comment = bytes(raw_comment).decode("utf-8", errors="ignore")

    return comment.split(",")

def reconstruct_masks(root_dest_dir: str) -> None:
    """
    Using a pre-existing UNet (trained with YOLO cropped images), it performs inference, and then uses the coordinates from the images (embedded and saved in metadata during yolo_cropped.py), to reconstruct the masks.
    Args:
        root_dest_dir (str): destination directory
    Globals used: PARAMS, SPLIT, WIDTHS, MODEL_PATH, OG_IMG_SIZE

    TODO: Optimize with batch inference and threadpoolexecutor for I/O tasks
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_cfg = PARAMS['dataloader']
    dataloader = SegmentationDataLoader(
        root_path= d_cfg['root_path'],
        image_dir=os.path.join("images"),
        mask_dir=os.path.join("labels"),
        image_size=d_cfg['image_size'],
        augmentation=False,
        subsample=1.0,
        batch_size=1,
        num_workers=d_cfg['num_workers'],
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
    )

    train, test = dataloader.get_dataloader()
    
    model = UNet(in_channels=PARAMS['model']['in_channels'], widths=WIDTHS, num_classes=PARAMS['model']['out_channels']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    dest_dir = os.path.join(root_dest_dir, SPLIT) ; create_dir(dest_dir)

    ### Counting Images with Metadata Detected
    ### ALL OF THEM SHOULD HAVE METADATA
    total_positive = total_negative = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test)):
            img = img_mask[0].float().to(device) # (1, 4, 128, 128) 
            image_path = test.dataset.image_dir + test.dataset.basenames[idx] + ".png"

            # Open and read Exif Metadata
            pil_img = Image.open(image_path)
            coords = read_metadata(pil_img)

            if coords != None:
                total_positive+=1

                pred_mask = torch.nn.functional.sigmoid(model(img)) # (1, 1, 128, 128)

                # Resize the predictions
                x1, y1, x2, y2 = coords
                height, width = abs(int(y1)-int(y2)), abs(int(x1)-int(x2))
                transform = transforms.Resize((height, width)) 
                pred_mask = transform(pred_mask).squeeze(0).squeeze(0) # (1, 1, H, W) where H and W where the original cropped images

                # Insert the predictions to full size empty mask
                full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device) 
                full_size_mask[int(y1):int(y2), int(x1):int(x2)] = pred_mask

                # Binarize the mask
                full_size_mask = (full_size_mask > 0.5).float()

                # Save the full size mask
                dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
                cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))
            else:
                total_negative+=1

                dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
                full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device)
                cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))

    print("\nTotal images with metadata: ", total_positive)
    print("Total images without metadata: ", total_negative)

    if total_negative > 1: 
        print(f"\nWARNING: Metadata not present in {total_negative} images")

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    Reconstruct Masks from YOLO Cropped Images by Using a Pre-Trained UNet
    Used for Evaluating Later
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--original_image_size", type=int, help='original image size to reconstruct towards\t[160]')
    parser.add_argument("-u", "--unet_image_size", type=int, help='image size used to train unet\t[128]')
    parser.add_argument('-w', '--widths', nargs='+', type=int, help='widths of the unet to reconstruct model', default=[64, 128, 256, 512])
    parser.add_argument("-m", "--model_path", type=str, help='path of pretrained unet model weights\t[checkpoints/unet_0/best.pt]')
    parser.add_argument("-d", "--data_path", type=str, help='root path of the dataset\t[3_fold_dataset/stacked_segmentation_0]')
    parser.add_argument("-s", "--split", type=str, help='split to evaluate\t[test]')
    parser.add_argument("-p", "--param_dir", type=str, help='directory of YAML parameter configuration file\t[parameters.yaml]')

    args = parser.parse_args()
    
    # Assign default values 
    OG_IMG_SIZE = args.original_image_size or 160
    UNET_IMG_SIZE = args.unet_image_size or 128
    WIDTHS = args.widths if len(args.widths) > 0 else [64, 128, 256, 512]
    MODEL_PATH = args.model_path or "checkpoints/unet_0/best.pt"
    SPLIT = args.split or "test"
    PARAM_DIR = args.param_dir or "parameters.yaml"

    with open(f"{PARAM_DIR}", "r") as f:
        PARAMS = yaml.safe_load(f)

    reconstruct_masks(f"reconstructed_{SPLIT}/labels")
