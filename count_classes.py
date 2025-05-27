import os
import glob
from collections import Counter
import argparse
import yaml

def count_class_instances(labels_dir, dataset_yaml_path=None):
    """
    Counts the occurrences of each class in YOLO format label files.

    Args:
        labels_dir (str): Path to the main directory containing 'train', 'val', etc., subdirectories
                          which in turn contain the .txt label files.
                          Example: 'path/to/dataset/labels'
        dataset_yaml_path (str, optional): Path to the dataset.yaml file to get class names.
                                           If None, only class indices will be shown.

    Returns:
        tuple: (class_counts_dict, total_instances)
               class_counts_dict: A dictionary where keys are class names (or indices)
                                  and values are their counts.
               total_instances: Total number of object instances found.
    """
    class_indices = []
    total_files_processed = 0
    total_instances = 0

    # Find all .txt files recursively within the labels_dir
    # This will cover train, val, test subdirectories if they exist
    for filepath in glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True):
        total_files_processed += 1
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts: # If line is not empty
                        try:
                            class_idx = int(parts[0])
                            class_indices.append(class_idx)
                            total_instances += 1
                        except ValueError:
                            print(f"Warning: Could not parse class index in {filepath} on line: '{line.strip()}'")
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

    if not class_indices and total_files_processed > 0:
        print(f"No valid class instances found in {total_files_processed} .txt files processed.")
    elif total_files_processed == 0:
        print(f"No .txt label files found in directory: {labels_dir}")
        return {}, 0

    class_counts = Counter(class_indices)
    
    class_names = None
    if dataset_yaml_path:
        try:
            with open(dataset_yaml_path, 'r') as f:
                dataset_info = yaml.safe_load(f)
                if 'names' in dataset_info:
                    class_names = dataset_info['names']
                else:
                    print(f"Warning: 'names' key not found in {dataset_yaml_path}.")
        except Exception as e:
            print(f"Error reading dataset YAML {dataset_yaml_path}: {e}")

    # Prepare dictionary with class names if available
    class_counts_dict = {}
    sorted_class_indices = sorted(class_counts.keys())

    if class_names:
        for idx in sorted_class_indices:
            name = class_names[idx] if 0 <= idx < len(class_names) else f"Class_{idx}"
            class_counts_dict[name] = class_counts[idx]
    else:
        for idx in sorted_class_indices:
            class_counts_dict[f"Class_{idx}"] = class_counts[idx]
            
    return class_counts_dict, total_instances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count class instances in YOLO label files.")
    parser.add_argument('--labels_dir', type=str, required=True,
                        help="Path to the root directory containing label subfolders (e.g., 'path/to/dataset/labels').")
    parser.add_argument('--dataset_yaml', type=str, default=None,
                        help="Optional: Path to the dataset.yaml file to get class names.")
    
    args = parser.parse_args()

    print(f"\nScanning label files in: {args.labels_dir}")
    if args.dataset_yaml:
        print(f"Using class names from: {args.dataset_yaml}")

    counts, total = count_class_instances(args.labels_dir, args.dataset_yaml)

    if counts:
        print("\n--- Class Instance Counts ---")
        max_name_len = max(len(name) for name in counts.keys()) if counts else 10
        for name, count in counts.items():
            print(f"{name:<{max_name_len}} : {count}")
        print("-----------------------------")
        print(f"{'Total Instances':<{max_name_len}} : {total}")
        print("-----------------------------")
    else:
        if total_files_processed > 0: # Check if files were processed but no instances found
             print("No class instances were successfully parsed from the label files.")