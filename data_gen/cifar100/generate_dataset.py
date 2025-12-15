from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import clip
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import tqdm
import copy
import os
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger
import json, math
import multiprocessing
import sys

# Configuration: modify this path according to your setup
DATA_ROOT = "./data/cifar100"

# 1. Data Preparation
class CIFAR100Subset(Dataset):
    def __init__(self, root, train=True, classes=None):
        super().__init__()
        full_dataset = CIFAR100(root, train=train, download=True)
        
        # Select specified classes
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Filter samples
        self.samples = [
            (img, self.class_to_idx[full_dataset.classes[label]])
            for img, label in full_dataset 
            if full_dataset.classes[label] in self.classes
        ]
        
        # CLIP official preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.preprocess(img), label

# 2. Model Definition
class CLIPAdapter(torch.nn.Module):

    def __init__(self, clip_model, hidden_dim=8):
        super().__init__()
        self.clip = clip_model
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
        train_dataset = CIFAR100Subset(
            root=DATA_ROOT,
            train=True,
            classes=selected_classes
        )
        val_dataset = CIFAR100Subset(
            root=DATA_ROOT,
            train=False,
            classes=selected_classes
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

    with open('./cifar100_classes.json', 'r') as f:
        classes = json.load(f)
    print(classes)
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))

    batch_size = 64
    adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    train_dataset = CIFAR100Subset(
        root=DATA_ROOT,
        train=True,
        classes=classes
    )
    val_dataset = CIFAR100Subset(
        root=DATA_ROOT,
        train=False,
        classes=classes
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
