# manual_strip.py
import torch
import argparse
from pathlib import Path
from utils.general import strip_optimizer # Make sure this path is correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manually strip optimizer from YOLOv7 checkpoint.")
    parser.add_argument('--weights', required=True, type=str, help='Path to the .pt checkpoint file to strip.')
    parser.add_argument('--out', type=str, default='', help='Path to save the stripped .pt file. Overwrites original if empty.')
    args = parser.parse_args()

    input_file = Path(args.weights)
    output_file = Path(args.out) if args.out else input_file # Overwrite if no output specified

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        exit()

    print(f"Stripping optimizer from: {input_file}")
    strip_optimizer(str(input_file), str(output_file)) # Call the utility function
    print(f"Stripped model saved to: {output_file}")
    print(f"Original size: {input_file.stat().st_size / (1024*1024):.2f} MB")
    if output_file.exists():
        print(f"New size: {output_file.stat().st_size / (1024*1024):.2f} MB")

'''
python manual_strip.py --weights path/to/best.pt --out best_stripped.pt
'''