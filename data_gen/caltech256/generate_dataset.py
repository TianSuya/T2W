from torchvision.datasets import Caltech256
import torch
import numpy as np
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import torchvision
import os
import clip
import tqdm
import copy
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import math,json,multiprocessing
import sys

def convert_to_rgb(image):
    """将单通道灰度图像转换为三通道RGB"""
    return image.convert('RGB')  # 对已经是RGB的图像无影响

def split_list(lst, ratios, num_splits):
    """
    将列表按照指定比例和数量拆分成子列表
    :param lst: 待拆分列表
    :param ratios: 每个子列表的元素占比，由小数表示的列表
    :param num_splits: 子列表的数量
    :return: 拆分后的子列表组成的列表
    """
    if len(ratios) != num_splits:
        raise ValueError("The length of ratios must equal to num_splits.")
    total_ratio = sum(ratios)
    if total_ratio != 1:
        raise ValueError("The sum of ratios must be equal to 1.")
    n = len(lst)
    result = []
    start = 0
    for i in range(num_splits):
        end = start + int(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result

class Caltech256Subset(torch.utils.data.Dataset):
    """
    动态过滤Caltech 256数据集的PyTorch Dataset
    功能：根据传入的类别名列表自动过滤样本，并重新映射标签
    
    参数：
        root (str): 数据集根目录
        class_names (list): 目标类别名列表 (e.g. ["ak47", "hedgehog"])
        transform (callable): 数据增强/预处理
        download (bool): 是否自动下载数据集
    """
    def __init__(self, root, class_names, split, download=False, seed=42):
        super().__init__()

        self.full_dataset = Caltech256(root=root, transform=None, download=download)
        self.class_names = class_names
        self.classes = class_names
        random.seed(seed)

        # 训练集数据增强（数据预处理）
        train_transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            # 随机缩放裁剪 (保持宽高比)
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机颜色扰动
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 转换为Tensor并自动归一化到[0,1]
            transforms.ToTensor(),
            # ImageNet标准归一化参数
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB均值
                                std=[0.229, 0.224, 0.225])   # RGB标准差
        ])

        # 测试集数据预处理
        test_transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            # 保持宽高比的缩放
            transforms.Resize(256),
            # 中心裁剪
            transforms.CenterCrop(224),
            # 转换为Tensor
            transforms.ToTensor(),
            # 相同归一化参数
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        if split == 'train':
            self.transform = train_transform
        else: self.transform = test_transform
        
        # 创建类别名到原始标签的映射
        self._create_label_mapping()
        
        # 生成过滤后的索引
        self.filtered_indices = self._filter_samples()
        random.shuffle(self.filtered_indices)
        self.train_indices, self.test_indices = split_list(self.filtered_indices, [0.8, 0.2], 2)
        if split == 'train':
            self.filtered_indices = self.train_indices
        else:
            self.filtered_indices = self.test_indices        

        # 验证是否找到有效样本
        if len(self.filtered_indices) == 0:
            raise ValueError("未找到与指定类别匹配的样本，请检查类别名称")

    def _create_label_mapping(self):
        """解析原始数据集的类别结构"""
        self.raw_categories = {}  # 原始类别名到标签的映射
        self.label_mapping = {}   # 新标签到原始标签的映射
        
        # 解析文件夹名称
        for idx, folder_name in enumerate(self.full_dataset.categories):
            _, name = folder_name.split(".", 1)
            self.raw_categories[name] = idx
            
        # 验证输入类别有效性
        valid_names = []
        for name in self.class_names:
            if name in self.raw_categories:
                valid_names.append(name)
            else:
                print(f"警告：类别 '{name}' 不存在，已自动过滤")
        
        # 创建新标签映射
        self.class_names = valid_names
        self.raw_labels = [self.raw_categories[name] for name in self.class_names]
        self.new_label_map = {raw: new for new, raw in enumerate(self.raw_labels)}

    def _filter_samples(self):
        """生成过滤后的样本索引"""
        indices = []
        for idx in range(len(self.full_dataset)):
            _, label = self.full_dataset[idx]
            if label in self.raw_labels:
                indices.append(idx)
        return indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, original_label = self.full_dataset[original_idx]
        image = self.transform(image)
        return image, torch.tensor(self.new_label_map[original_label])

    @property
    def class_stats(self):
        """获取类别统计信息"""
        counts = np.zeros(len(self.class_names))
        for _, label in self:
            counts[label] += 1
        return {name: int(counts[i]) for i, name in enumerate(self.class_names)}

class CLIPAdapter(torch.nn.Module):

    def __init__(self, clip_model=None, hidden_dim=8):
        super().__init__()
        self.clip = clip_model
        if clip_model is not None:
            self.clip.requires_grad_(False)  # 冻结CLIP
        self.resnet = torchvision.models.resnet18(pretrained=True)
        for name, param in self.resnet.named_parameters():
            param.requires_grad = False
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(512, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 512)
        )
    
    def encode_text(self, text_inputs):
        with torch.no_grad():
            init_text_features = self.clip.encode_text(text_inputs)
            init_text_features /= init_text_features.norm(dim=-1, keepdim=True)
            text_features = init_text_features.detach().float()
        return text_features
    
    def encode_image(self, image_inputs):
        with torch.no_grad():
            image_features = self.clip.encode_image(image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach().float()
        return image_features


    def forward(self, images):
        
        adapted_features = self.resnet(images)
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)
        
        return {
            "adapted": adapted_features,
        }

# 3. 训练函数
def train(model, train_loader, test_loader, optimizer, epochs, save_path, device):
    
    # 生成文本分类器（Zero-Shot基准）
    model = model.to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_loader.dataset.classes]).to(device)
    text_features = model.encode_text(text_inputs)
    test(model, test_loader, device, save_path)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            adapted_features = outputs["adapted"]
            
            # 任务损失
            logits_scale = model.clip.logit_scale.exp()
            logits = logits_scale * adapted_features @ text_features.T
            loss_task = F.cross_entropy(logits, labels)
            loss = loss_task
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
        test(model, test_loader, device, save_path)

# 4. 测试函数
def test(model, test_loader, device, save_path):
    model.eval()
    correct = 0
    total = 0
    
    # 生成文本分类器
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_loader.dataset.classes]).to(device)
    text_features = model.encode_text(text_inputs)
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            features = outputs['adapted']
            
            logits_scale = model.clip.logit_scale.exp()
            logits = logits_scale * features @ text_features.T
            _, predicted = torch.max(logits, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")

    clip_adapter = copy.deepcopy(model.resnet.fc.state_dict())
    save_dict = {
        'clip_adapter':clip_adapter,
        'accuracy':acc
    }
    torch.save(save_dict, save_path)

    return acc

def train_task():
    with open('./data_point.json', 'r') as f:
        data_point = json.load(f)

    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    ckpt_path = './checkpoints/base.pt'
    data_root = './data'
    batch_size = 64
    resume_ckpt = torch.load(ckpt_path)
    ckpt = resume_ckpt['clip_adapter']
    acc = resume_ckpt['accuracy']
    print(f'Resume from {ckpt_path}, model accuracy is:{acc}')
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    init_adapter.resnet.fc.load_state_dict(ckpt)

    for item in data_point[begin:end]:
        index = item['index']
        selected_classes = item['selected_classes']
        train_dataset = Caltech256Subset(
            root=data_root,
            class_names=selected_classes,
            split='train',
        )
        val_dataset = Caltech256Subset(
            root=data_root,
            class_names=selected_classes,
            split='test',
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        # 创建检查点目录
        pt_save_path = os.path.join('./checkpoints', str(index))
        os.makedirs(pt_save_path, exist_ok=True)
        save_path = os.path.join(pt_save_path, 'task.pt')
        adapter = copy.deepcopy(init_adapter)
        optim = torch.optim.Adam(adapter.resnet.fc.parameters(), lr=1e-4)

        train(
            model=adapter,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optim,
            epochs=10,
            save_path=save_path,
            device=torch.device(f'cuda:{device_num}')
        )

def train_base():

    with open('./caltech_classes.json', 'r') as f:
        classes = json.load(f)
    print(classes)
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    data_root = './data'
    batch_size = 64
    adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    train_dataset = Caltech256Subset(
        root=data_root,
        class_names=classes,
        split='train',
    )
    val_dataset = Caltech256Subset(
        root=data_root,
        class_names=classes,
        split='test',
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # 创建检查点目录
    pt_save_path = os.path.join('./checkpoints')
    save_path = os.path.join(pt_save_path, 'base-fusion.pt')
    optim = torch.optim.Adam(adapter.resnet.fc.parameters(), lr=1e-5)

    train(
        model=adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        epochs=1,
        save_path=save_path,
        device=torch.device(f'cuda:{device_num}')
    )
    


if __name__ == '__main__':
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    device_num = int(sys.argv[3])
    # train_task()
    train_base()