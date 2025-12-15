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

# Configuration: modify this path according to your setup
DATA_ROOT = "./data"

def convert_to_rgb(image):
    """Convert single-channel grayscale image to three-channel RGB"""
    return image.convert('RGB')  # No effect on images already in RGB

def split_list(lst, ratios, num_splits):
    """
    Split a list into sublists according to specified ratios.
    :param lst: List to be split
    :param ratios: Ratio for each sublist, represented as decimals
    :param num_splits: Number of sublists
    :return: List of sublists
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
    PyTorch Dataset for dynamically filtering Caltech 256 dataset.
    Function: Automatically filter samples based on the provided class name list and remap labels.
    
    Args:
        root (str): Dataset root directory
        class_names (list): List of target class names (e.g. ["ak47", "hedgehog"])
        transform (callable): Data augmentation/preprocessing
        download (bool): Whether to automatically download the dataset
    """
    def __init__(self, root, class_names, split, download=False, seed=42):
        super().__init__()

        self.full_dataset = Caltech256(root=root, transform=None, download=download)
        self.class_names = class_names
        self.classes = class_names
        random.seed(seed)

        # Training data augmentation (data preprocessing)
        train_transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            # Random resize crop (maintain aspect ratio)
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            # Random color jitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # Convert to Tensor and normalize to [0,1]
            transforms.ToTensor(),
            # ImageNet standard normalization parameters
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB mean
                                std=[0.229, 0.224, 0.225])   # RGB std
        ])

        # Test data preprocessing
        test_transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            # Resize maintaining aspect ratio
            transforms.Resize(256),
            # Center crop
            transforms.CenterCrop(224),
            # Convert to Tensor
            transforms.ToTensor(),
            # Same normalization parameters
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        if split == 'train':
            self.transform = train_transform
        else: self.transform = test_transform
        
        # Create class name to original label mapping
        self._create_label_mapping()
        
        # Generate filtered indices
        self.filtered_indices = self._filter_samples()
        random.shuffle(self.filtered_indices)
        self.train_indices, self.test_indices = split_list(self.filtered_indices, [0.8, 0.2], 2)
        if split == 'train':
            self.filtered_indices = self.train_indices
        else:
            self.filtered_indices = self.test_indices        

        # Validate if valid samples are found
        if len(self.filtered_indices) == 0:
            raise ValueError("No samples matching the specified classes were found. Please check class names.")

    def _create_label_mapping(self):
        """Parse the original dataset's class structure"""
        self.raw_categories = {}  # Original class name to label mapping
        self.label_mapping = {}   # New label to original label mapping
        
        # Parse folder names
        for idx, folder_name in enumerate(self.full_dataset.categories):
            _, name = folder_name.split(".", 1)
            self.raw_categories[name] = idx
            
        # Validate input class validity
        valid_names = []
        for name in self.class_names:
            if name in self.raw_categories:
                valid_names.append(name)
            else:
                print(f"Warning: Class '{name}' does not exist, automatically filtered")
        
        # Create new label mapping
        self.class_names = valid_names
        self.raw_labels = [self.raw_categories[name] for name in self.class_names]
        self.new_label_map = {raw: new for new, raw in enumerate(self.raw_labels)}

    def _filter_samples(self):
        """Generate filtered sample indices"""
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
        """Get class statistics"""
        counts = np.zeros(len(self.class_names))
        for _, label in self:
            counts[label] += 1
        return {name: int(counts[i]) for i, name in enumerate(self.class_names)}

class CLIPAdapter(torch.nn.Module):

    def __init__(self, clip_model=None, hidden_dim=8):
        super().__init__()
        self.clip = clip_model
        if clip_model is not None:
            self.clip.requires_grad_(False)  # Freeze CLIP
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

# 3. Training Function
def train(model, train_loader, test_loader, optimizer, epochs, save_path, device):
    
    # Generate text classifier (Zero-Shot baseline)
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
            
            # Task loss
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

# 4. Testing Function
def test(model, test_loader, device, save_path):
    model.eval()
    correct = 0
    total = 0
    
    # Generate text classifier
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
            root=DATA_ROOT,
            class_names=selected_classes,
            split='train',
        )
        val_dataset = Caltech256Subset(
            root=DATA_ROOT,
            class_names=selected_classes,
            split='test',
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        # Create checkpoint directory
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

    batch_size = 64
    adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    train_dataset = Caltech256Subset(
        root=DATA_ROOT,
        class_names=classes,
        split='train',
    )
    val_dataset = Caltech256Subset(
        root=DATA_ROOT,
        class_names=classes,
        split='test',
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # Create checkpoint directory
    pt_save_path = os.path.join('./checkpoints')
    os.makedirs(pt_save_path, exist_ok=True)
    save_path = os.path.join(pt_save_path, 'base.pt')
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
