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
import random
from torch import nn
from scipy.optimize import linear_sum_assignment

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

def initialize_weights(init_type, module_list):
    for m in module_list:
        if type(m) == nn.Linear:
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform(m.weight)
            if init_type == "xavier_normal":
                nn.init.xavier_normal(m.weight)
            if init_type == "uniform":
                nn.init.uniform(m.weight)
            if init_type == "normal":
                nn.init.normal(m.weight)
            if init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform(m.weight)
            m.bias.data.fill_(0.01)
    return module_list

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
    total_loss = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            features = outputs['adapted']
            
            logits_scale = model.clip.logit_scale.exp()
            logits = logits_scale * features @ text_features.T
            loss = F.cross_entropy(logits, labels)
            total_loss.append(loss)
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
    if save_path is not None:
        torch.save(save_dict, save_path)

    return acc, sum(total_loss)/len(total_loss)

def fusion(data_points):

    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    batch_size = 64
    init_adapter = resume_model('./checkpoints/base-fusion.pt', clip_model)

    count = 0

    for item in data_points:
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
        if count == 0: save_name = 'modelA.pt'
        else: save_name = 'modelB.pt'
        pt_save_path = os.path.join('./fusion')
        os.makedirs(pt_save_path, exist_ok=True)
        save_path = os.path.join(pt_save_path, save_name)
        adapter = copy.deepcopy(init_adapter)
        optim = torch.optim.Adam(adapter.resnet.fc.parameters(), lr=1e-3)

        train(
            model=adapter,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optim,
            epochs=10,
            save_path=save_path,
            device=torch.device(f'cuda:{device_num}')
        )

        count += 1

def resume_model(path, clip_model):
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    resume_ckpt = torch.load(path)
    ckpt = resume_ckpt['clip_adapter']
    acc = resume_ckpt['accuracy']
    init_adapter.resnet.fc.load_state_dict(ckpt)
    return init_adapter

def linear_interpolate_heads(head_a, head_b):
    """
    直接线性插值两个分类头（无排列对齐）
    参数：
        head_a, head_b: 两个结构相同的nn.Sequential分类头
    返回：
        插值后的新分类头（参数 = 0.5*head_a + 0.5*head_b）
    """
    # 确保设备一致
    device = head_a[0].weight.device
    
    # 初始化新分类头
    fused_head = torch.nn.Sequential(
        torch.nn.Linear(512, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 512)
    ).to(device)

    # 逐层参数插值
    with torch.no_grad():
        # 第一个线性层（512->16）
        fused_head[0].weight.data = 0.5 * head_a[0].weight + 0.5 * head_b[0].weight
        fused_head[0].bias.data = 0.5 * head_a[0].bias + 0.5 * head_b[0].bias
        
        # 第二个线性层（16->512）
        fused_head[2].weight.data = 0.5 * head_a[2].weight + 0.5 * head_b[2].weight
        fused_head[2].bias.data = 0.5 * head_a[2].bias + 0.5 * head_b[2].bias

    return fused_head

def rebasin_fusion(head_a, head_b):
    """
    融合两个CLIP分类头，使用Git Re-basin方法对齐隐藏层神经元排列
    参数：
        head_a, head_b: 两个nn.Sequential分类头，结构为[Linear(512,16), GELU, Linear(16,512)]
    返回：
        融合后的新分类头
    """
    # 确保设备一致
    device = head_a[0].weight.device
    
    # 初始化目标分类头
    fused_head = torch.nn.Sequential(
        torch.nn.Linear(512, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 512)
    ).to(device)

    # 阶段1：对齐第一个线性层（512->16）
    # ------------------------------------
    # 获取权重矩阵（形状[16, 512]）
    W_a1 = head_a[0].weight.data.cpu().numpy()
    W_b1 = head_b[0].weight.data.cpu().numpy()
    
    # 计算排列映射（使用匈牙利算法）
    cost_matrix = np.linalg.norm(W_a1[:, None] - W_b1[None, :], axis=2)  # 形状[16, 16]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    permutation = col_ind  # 将B的神经元映射到A的顺序

    print(permutation)

    # 应用排列到B的第一个线性层
    aligned_W_b1 = W_b1[permutation]
    aligned_bias_b1 = head_b[0].bias.data.cpu().numpy()[permutation]

    # 阶段2：对齐第二个线性层（16->512）
    # ------------------------------------
    # 获取B的第二个线性层权重（形状[512,16]）
    W_b2 = head_b[2].weight.data.cpu().numpy()
    
    # 根据第一阶段排列调整输入通道
    aligned_W_b2 = W_b2[:, permutation]  # 调整输入维度排列

    # 阶段3：参数融合
    # ------------------------------------
    # 第一个线性层融合（平均）
    fused_W1 = (head_a[0].weight.data + torch.from_numpy(aligned_W_b1).to(device)) / 2
    fused_bias1 = (head_a[0].bias.data + torch.from_numpy(aligned_bias_b1).to(device)) / 2
    
    # 第二个线性层融合（平均）
    fused_W2 = (head_a[2].weight.data + torch.from_numpy(aligned_W_b2).to(device)) / 2
    fused_bias2 = (head_a[2].bias.data + head_b[2].bias.data) / 2  # 直接平均输出偏置

    # 参数赋值
    fused_head[0].weight.data.copy_(fused_W1)
    fused_head[0].bias.data.copy_(fused_bias1)
    fused_head[2].weight.data.copy_(fused_W2)
    fused_head[2].bias.data.copy_(fused_bias2)

    return fused_head

def eval_merge(data_points):
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    batch_size = 64
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    model_A_ckpt_path = './fusion/modelA.pt'
    model_B_ckpt_path = './fusion/modelB.pt'

    model_A = resume_model(model_A_ckpt_path, clip_model).cuda(device_num)
    model_B = resume_model(model_B_ckpt_path, clip_model).cuda(device_num)

    count = 0

    selected_classes_A = data_points[0]['selected_classes']
    val_dataset_A = TinyImageNet(
        root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
        train=False,
        transform=train_transform,
        class_names=selected_classes_A
    )
    test_loader_A = DataLoader(val_dataset_A, batch_size=batch_size, shuffle=False, num_workers=16)

    selected_classes_B = data_points[1]['selected_classes']
    val_dataset_B = TinyImageNet(
        root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
        train=False,
        transform=train_transform,
        class_names=selected_classes_B
    )
    test_loader_B = DataLoader(val_dataset_B, batch_size=batch_size, shuffle=False, num_workers=16)

    test(model_A, test_loader_A, device=device_num, save_path=None)
    test(model_A, test_loader_B, device=device_num, save_path=None)
    test(model_B, test_loader_A, device=device_num, save_path=None)
    test(model_B, test_loader_B, device=device_num, save_path=None)

    rebasin_model = copy.deepcopy(init_adapter)
    rebasin_head = rebasin_fusion(model_A.resnet.fc, model_B.resnet.fc)
    rebasin_model.resnet.fc = rebasin_head
    rebasin_model = rebasin_model.cuda(device_num)

    test(rebasin_model, test_loader_A, device=device_num, save_path=None)
    test(rebasin_model, test_loader_B, device=device_num, save_path=None)

    interpolate_model = copy.deepcopy(init_adapter)
    interpolate_head = linear_interpolate_heads(model_A.resnet.fc, model_B.resnet.fc)
    interpolate_model.resnet.fc = interpolate_head
    interpolate_model = interpolate_model.cuda(device_num)

    test(interpolate_model, test_loader_A, device=device_num, save_path=None)
    test(interpolate_model, test_loader_B, device=device_num, save_path=None)

    t2w_model = resume_model('/data/bowen/Text2Weight/imagenet_fusion.pt', clip_model=clip_model).cuda(device_num)
    test(t2w_model, test_loader_A, device=device_num, save_path=None)
    test(t2w_model, test_loader_B, device=device_num, save_path=None)


def get_point():

    with open('./val.json','r') as f:
        data_points = json.load(f)
    
    res = None
    for item in data_points:
        if len(item['selected_classes']) == 16:
            res = item
            break

    data_point_A = {}
    data_point_B = {}

    random.seed(42)

    classes_A = random.choices(res['selected_classes'],k=8)
    classes_B = [c for c in res['selected_classes'] if c not in classes_A]

    data_point_A['index'] = 0
    data_point_B['index'] = 1
    data_point_A['selected_classes'] = classes_A
    data_point_B['selected_classes'] = classes_B

    result = [data_point_A, data_point_B]

    with open('./fusion_data.json','w') as f:
        json.dump(result, f)

    with open('./selected_point.json','w') as f:
        json.dump(res, f)
    
    return result

def train_t2w(selected_point):
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    t2w_model = resume_model('/data/bowen/Text2Weight/imagenet_fusion.pt', clip_model=clip_model).cuda(device_num)
    batch_size = 64
    data_root = './data'
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )

    selected_classes = selected_point['selected_classes']
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
    now_model = t2w_model
    save_name = 't2w'

    pt_save_path = os.path.join('./init')
    os.makedirs(pt_save_path, exist_ok=True)
    save_path = os.path.join(pt_save_path, save_name)
    optim = torch.optim.Adam(now_model.resnet.fc.parameters(), lr=1e-4)

    train(
        model=now_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        epochs=50,
        save_path=save_path,
        device=torch.device(f'cuda:{device_num}')
    )

def train_initialize(selected_point):
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    # t2w_model = resume_model('/data/bowen/Text2Weight/cifar_fusion.pt', clip_model=clip_model).cuda(device_num)
    batch_size = 64
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )

    selected_classes = selected_point['selected_classes']
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

    initialize_list = ['xavier_uniform', 'xavier_normal', 'uniform', 'normal', 'kaiming_normal', 'kaiming_uniform']
    for init in initialize_list:
        now_model = copy.deepcopy(init_adapter)
        save_name = init
        initialize_weights(init, now_model.resnet.fc)

        pt_save_path = os.path.join('./init')
        os.makedirs(pt_save_path, exist_ok=True)
        save_path = os.path.join(pt_save_path, save_name)
        optim = torch.optim.Adam(now_model.resnet.fc.parameters(), lr=1e-4)

        train(
            model=now_model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optim,
            epochs=50,
            save_path=save_path,
            device=torch.device(f'cuda:{device_num}')
        )

def eval_init(selected_point):
    data_root = './data'
    clip_model, _ = clip.load("ViT-B/32", device=torch.device(f"cuda:{device_num}"))
    batch_size = 64
    init_adapter = CLIPAdapter(
        clip_model=clip_model,
        hidden_dim=16
    )
    selected_classes = selected_point['selected_classes']
    val_dataset = TinyImageNet(
        root='/data/bowen/Text2Weight/data/tiny-imagenet-200',
        train=False,
        transform=train_transform,
        class_names=selected_classes
    )
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    model_list = ['xavier_uniform', 'xavier_normal', 'uniform', 'normal', 'kaiming_normal', 'kaiming_uniform', 't2w']
    for model_name in model_list:
        now_model = resume_model('./init/'+model_name, clip_model).cuda(device_num)
        acc, loss = test(now_model, test_loader, device=device_num, save_path=None)
        print(f'model:{model_name}, accuracy:{acc}, loss:{loss}')

if __name__ == '__main__':
    device_num = int(sys.argv[1])
    # points = get_point()
    with open('./selected_point.json', 'r') as f:
        points = json.load(f)
    # train_t2w(points)
    eval_init(points)
    # fusion(points)