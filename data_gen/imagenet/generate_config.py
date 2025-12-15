import json
from torchvision.datasets import CIFAR100
import numpy as np
from pathlib import Path
np.random.seed(42)

with open('./imagenet_classes.json','r') as f:
    imagenet_classes = json.load(f)

data_point = 12000

class_num_range = [i for i in range(8,32)]

total_info = []

for item in range(data_point):
    data_id = item
    num_class = list(np.random.choice(class_num_range, 1, replace=False))[0]
    # print(num_class)
    selected_classes = list(np.random.choice(imagenet_classes, num_class, replace=False))
    data_info = {
        'index': data_id,
        'selected_classes': selected_classes,
    }
    total_info.append(data_info)

with open('./data_point.json', 'w') as f:
    json.dump(total_info, f)