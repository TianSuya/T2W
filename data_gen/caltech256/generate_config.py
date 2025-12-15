import json
from torchvision.datasets import CIFAR100
import numpy as np
from pathlib import Path
np.random.seed(42)

# Configuration: modify this path according to your setup
CALTECH256_ROOT = "./data/caltech256/256_ObjectCategories"

def get_caltech256_class_names(root_dir):
    """
    Get all class names from Caltech 256 dataset.
    
    Args:
        root_dir (str): Root directory path of the dataset (containing numbered class folders)
    
    Returns:
        list: List of class names sorted by number
    
    Raises:
        ValueError: When invalid directory naming format is detected
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")
    
    classes = []
    invalid_dirs = []
    
    # Iterate through all subdirectories
    for dir_path in root.iterdir():
        if dir_path.is_dir():
            dir_name = dir_path.name
            # Split directory name
            if '.' not in dir_name:
                invalid_dirs.append(dir_name)
                continue
                
            parts = dir_name.split('.', 1)
            prefix, class_name = parts[0], parts[1]
            
            # Validate prefix format
            if len(prefix) != 3 or not prefix.isdigit():
                invalid_dirs.append(dir_name)
                continue
                
            # Record valid class
            classes.append( (int(prefix), class_name) )
    
    # Handle invalid directories
    if invalid_dirs:
        raise ValueError(f"Found {len(invalid_dirs)} directories with invalid naming format, examples: {invalid_dirs[:3]}...")
    
    # Sort by number
    classes.sort(key=lambda x: x[0])
    
    return [class_name for _, class_name in classes]

caltech_classes = get_caltech256_class_names(root_dir=CALTECH256_ROOT)

with open('caltech_classes.json', 'w') as f:
    json.dump(caltech_classes, f)

data_point = 12000

class_num_range = [i for i in range(8,32)]

total_info = []

for item in range(data_point):
    data_id = item
    num_class = list(np.random.choice(class_num_range, 1, replace=False))[0]
    # print(num_class)
    selected_classes = list(np.random.choice(caltech_classes, num_class, replace=False))
    data_info = {
        'index': data_id,
        'selected_classes': selected_classes,
    }
    total_info.append(data_info)

with open('./data_point.json', 'w') as f:
    json.dump(total_info, f)



