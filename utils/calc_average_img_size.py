import os
from PIL import Image
from collections import Counter
import statistics

def analyze_image_dimensions(directory_path):
    """Analyze PNG image dimensions and calculate mean, median, mode."""
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist.")

    dimensions = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    dimensions.append((width, height))
            except Exception as e:
                print(f"Warning: Could not process {filename}: {e}")

    if not dimensions:
        raise ValueError("No PNG images found.")

    # Extract widths and heights
    widths, heights = zip(*dimensions)

    # Calculate statistics
    mean_width = statistics.mean(widths)
    mean_height = statistics.mean(heights)
    median_width = statistics.median(widths)
    median_height = statistics.median(heights)
    mode_width = statistics.mode(widths)
    mode_height = statistics.mode(heights)

    return (mean_width, mean_height), (median_width, median_height), (mode_width, mode_height)

def main():
    directory_path = input("Enter directory path: ").strip()
    
    try:
        mean_dim, median_dim, mode_dim = analyze_image_dimensions(directory_path)
        print(f"Mean: {mean_dim[0]:.2f} x {mean_dim[1]:.2f}")
        print(f"Median: {median_dim[0]} x {median_dim[1]}")
        print(f"Mode: {mode_dim[0]} x {mode_dim[1]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
