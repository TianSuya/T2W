import json
from torchvision.datasets import CIFAR100
import numpy as np
from pathlib import Path
np.random.seed(42)

def get_caltech256_class_names(root_dir):
    """
    获取Caltech 256数据集的所有类别名称
    
    参数:
        root_dir (str): 数据集根目录路径（包含编号类别文件夹的目录）
    
    返回:
        list: 按编号排序的类别名称列表
    
    异常:
        ValueError: 当检测到无效的目录命名格式时抛出
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")
    
    classes = []
    invalid_dirs = []
    
    # 遍历所有子目录
    for dir_path in root.iterdir():
        if dir_path.is_dir():
            dir_name = dir_path.name
            # 分割目录名
            if '.' not in dir_name:
                invalid_dirs.append(dir_name)
                continue
                
            parts = dir_name.split('.', 1)
            prefix, class_name = parts[0], parts[1]
            
            # 验证前缀格式
            if len(prefix) != 3 or not prefix.isdigit():
                invalid_dirs.append(dir_name)
                continue
                
            # 记录有效类别
            classes.append( (int(prefix), class_name) )
    
    # 处理无效目录
    if invalid_dirs:
        raise ValueError(f"发现{len(invalid_dirs)}个无效目录命名格式，示例: {invalid_dirs[:3]}...")
    
    # 按编号排序
    classes.sort(key=lambda x: x[0])
    
    return [class_name for _, class_name in classes]

caltech_classes = get_caltech256_class_names(root_dir='/data/bowen/Text2Weight/data_gen/caltech256/data/caltech256/256_ObjectCategories')

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



