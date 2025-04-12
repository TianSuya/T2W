import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
import clip, torchvision
import tqdm
import torch.nn.functional as F
import copy

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

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

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, class_names=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.class_names = class_names  # 新增参数

        # 初始化基础数据结构
        if not self.Train:
            self._create_class_idx_dict_val()

        # 加载类别名称映射
        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")
        
        self.set_nids = set()
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        self.label_to_class = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = words[1].strip("\n").split(",")[0]
                    self.label_to_class[words[1].strip("\n").split(",")[0]] = words[0]

        self.classes = class_names

        # 获取有效wnid列表
        self.selected_wnids = [self.label_to_class[name] for name in self.class_names]
        # 重新构建映射关系
        self.class_to_tgt_idx = {wnid: idx for idx, wnid in enumerate(self.selected_wnids)}
        self.tgt_idx_to_class = {idx: wnid for idx, wnid in enumerate(self.selected_wnids)}

        # 构建数据集
        self._make_dataset(self.Train)

    def _create_class_idx_dict_val(self):
        # 原始方法保持不变
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        
        self.val_img_to_class = {}
        with open(val_annotations_file, 'r') as fo:
            for line in fo.readlines():
                words = line.split("\t")
                self.val_img_to_class[words[0]] = words[1]

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            # 根据是否过滤类别选择目录
            if self.class_names is not None:
                list_of_dirs = self.selected_wnids
            else:
                list_of_dirs = list(self.class_to_tgt_idx.keys())
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in os.walk(dirs):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        
                        # 获取当前样本的wnid
                        if Train:
                            wnid = tgt
                        else:
                            wnid = self.val_img_to_class[fname]
                        
                        # 类别过滤逻辑
                        if self.class_names is not None:
                            if wnid not in self.selected_wnids:
                                continue
                        
                        # 获取对应标签
                        label = self.class_to_tgt_idx[wnid]
                        self.images.append((path, label))
        
        # 更新数据集长度
        self.len_dataset = len(self.images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]
    
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
        train_dataset = TinyImageNet(
            root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
            train=True,
            transform=train_transform,
            class_names=selected_classes
        )
        val_dataset = TinyImageNet(
            root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
            train=False,
            transform=train_transform,
            class_names=selected_classes
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

    with open('./imagenet_classes.json', 'r') as f:
        classes = json.load(f)
    print(classes)
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))

    batch_size = 64
    adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    train_dataset = TinyImageNet(
        root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
        train=True,
        transform=train_transform,
        class_names=classes
    )
    val_dataset = TinyImageNet(
        root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
        train=False,
        transform=train_transform,
        class_names=classes
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
