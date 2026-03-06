from typing import List
import subprocess 
import argparse
import time
import os
    
def run_subprocess(): 
    """
    Runs a script K times with different parameters   
    """

    command = ["python3", f"{SCRIPT}"]

    # Check if directories exist before running the command
    # ARGS = [ [LIST_0], [LIST_1], [LIST_2]]
    for i in range(K): 
        in_dir = ARGS[i][1]
        if os.path.exists(in_dir): 
            print("Directory exists: ", in_dir)
        else: 
            print("Directory does not exist: ", in_dir)
            return

    for i in range(K): 
        print(f"Running fold {i+1}/{K}...")

        # Expand the command to include the parameter directory
        k_command = command + ARGS[i]
        result = subprocess.run(k_command, text=True)

        # Print the result
        print("Output: ", result.stdout) # Output from the script
        print("\nError: ", result.stderr)  # Error from the script

        # Buffer time
        time.sleep(5)
        print(f"\nFinished fold {i+1}/{K}...")
        print("Waiting for 5 seconds before starting the next fold...\n")
        
if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    Run subprocess that calls a Python script 'K' times with custom argparse parameters
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--script", type=str,help='Script to run K times\t[yolo_crop.py]')
    parser.add_argument("-k", "--k", type=int,help='Run a script K times\t[3]')
    parser.add_argument('-a', '--args', nargs='+', help='Additional arguments to pass to the script', default=[])
    args = parser.parse_args()

    if args.k is not None:
        K = args.k
    else: K = 3
    if args.script is not None:
        SCRIPT = args.script
    else: SCRIPT = "yolo_crop.py"
    if args.args is not None: 
        ARGS = args.args
    else: 
        # For Running 'yolo_crop.py'
        ARGS = [
            [
            "--in_dir", '3_fold_dataset/stacked_segmentation_0', # UPDATE
            "--model_dir", 'pretrained_yolo/best_0.pt', # UPDATE
            "--device", 'cuda', 
            '--batch_size', '16',
            '--image_size', '160',
            '--confidence', '0.70',
            '--margin_of_error', '30',
            '--workers', '8',

            ],
            [
            "--in_dir", '3_fold_dataset/stacked_segmentation_1', # UPDATE
            "--model_dir", 'pretrained_yolo/best_1.pt', # UPDATE
            "--device", 'cuda', 
            '--batch_size', '16',
            '--image_size', '160',
            '--confidence', '0.70',
            '--margin_of_error', '30',
            '--workers', '8',
            ],
                        [
            "--in_dir", '3_fold_dataset/stacked_segmentation_2', # UPDATE
            "--model_dir", 'pretrained_yolo/best_2.pt', # UPDATE
            "--device", 'cuda', 
            '--batch_size', '16',
            '--image_size', '160',
            '--confidence', '0.70',
            '--margin_of_error', '30',
            '--workers', '8',
            ],
        ]

    run_subprocess()
    print(f"\nFinished Subprocess!")