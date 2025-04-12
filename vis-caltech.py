#!/usr/bin/env python3
"""
Script for training and evaluating G.pt models.
"""
try:
    import isaacgym
except ImportError:
    print("WARNING: Isaac Gym not imported")
import clip
import tqdm
import multiprocessing
import os
import hydra
import omegaconf
import random
from copy import deepcopy
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from Gpt.diffusion import create_diffusion
from Gpt.diffusion.timestep_sampler import UniformSampler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from data_gen.cifar100.p_dataset import PDataset
from Gpt.distributed import scaled_all_reduce
from Gpt.models.transformer import Gpt
import joblib
from Gpt.meters import TrainMeter, TestMeter
from Gpt.utils import setup_env, construct_loader, shuffle, update_lr, spread_losses, accumulate, requires_grad
from Gpt.distributed import get_rank, get_world_size, is_main_proc, synchronize
# from Gpt.vis import VisMonitor
# from Gpt.tasks import get
from Gpt.download import find_model
from data_gen.cifar100.generate_dataset import CIFAR100Subset
import numpy as np
from matplotlib.colors import Normalize
import scienceplots
import json
from data_gen.caltech256.generate_dataset import CLIPAdapter, Caltech256Subset

clip_model,_ = clip.load("ViT-B/32", device='cuda')
# plt.style.use(['science','no-latex'])

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def flatten_tensor_list(tensor_list):
    """将张量列表展平为单个向量"""
    return torch.cat([t.reshape(-1) for t in tensor_list])

def reconstruct_tensor_list(flat_tensor, original_shapes):
    """将展平的向量恢复为原始形状的张量列表"""
    tensor_list = []
    ptr = 0
    for shape in original_shapes:
        numel = int(np.prod(shape))
        print(numel)
        tensor_list.append(flat_tensor[ptr:ptr+numel].view(shape))
        ptr += numel
    return tensor_list

def visualize_2d_loss_landscape(
    model,              # 预训练模型（需要支持参数访问）
    initial_params,     # 初始参数（θ0）
    delta1,             # 优化方向1（θ1 - θ0）
    delta2,             # 优化方向2（θ2 - θ0）
    criterion,          # 损失函数
    dataloader,         # 数据加载器
    alpha_range=(-4, 4),# α范围
    beta_range=(-4, 4), # β范围
    num_points=40,      # 每个轴的采样点数
    device='cuda',      # 计算设备
    plot_type='3d',       # 新增绘图类型参数
    cmap='viridis',       # 新增颜色映射参数
    elevation=25,         # 3D视角仰角
    azimuth=45,            # 3D视角方位角
    save_name='test.png',
    data_path='data.pt'
):
    """
    绘制二维损失表面
    
    参数说明：
    - delta1：优化方向1（目标任务的参数变化量）
    - delta2：优化方向2（其他任务的参数变化量）
    - 返回值：网格化的损失值矩阵
    """
    
    # 确保模型参数与初始参数一致
    with torch.no_grad():
        for (name, param), init_val in zip(model.named_parameters(), initial_params):
            param.copy_(init_val)
    
    original_shapes = [t.shape for t in delta1]
    
    # 将张量列表展平为向量
    delta1_flat = flatten_tensor_list(delta1)
    delta2_flat = flatten_tensor_list(delta2)
    
    # 归一化处理（关键修正）
    delta1_norm = torch.norm(delta1_flat)
    delta2_flat = delta2_flat * (delta1_norm / torch.norm(delta2_flat))
    
    # 重建为原始形状的张量列表
    delta1 = reconstruct_tensor_list(delta1_flat, original_shapes)
    delta2 = reconstruct_tensor_list(delta2_flat, original_shapes)
    
    # 创建参数网格
    alpha_vals = np.linspace(*alpha_range, num_points)
    beta_vals = np.linspace(*beta_range, num_points)
    loss_grid = np.zeros((len(alpha_vals), len(beta_vals)))
    
    # 参数访问辅助函数
    def set_parameters(alpha, beta):
        with torch.no_grad():
            for param, init_val, d1, d2 in zip(model.parameters(), initial_params, delta1, delta2):
                new_val = init_val + alpha * d1 + beta * d2
                param.copy_(new_val)

    if os.path.exists(data_path):
        print('Reload From Data')
        loss_grid = torch.load(data_path, map_location='cpu')
    else:
    
        # 遍历整个参数空间
        for i, alpha in enumerate(tqdm.tqdm(alpha_vals, desc="Computing Loss Landscape")):
            for j, beta in enumerate(beta_vals):
                # 设置当前参数
                set_parameters(alpha, beta)
                
                # 计算损失
                model.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item() * inputs.size(0)
                
                # 记录平均损失
                loss_grid[i, j] = total_loss / len(dataloader.dataset)

        print("Data Saved")
        torch.save(loss_grid, data_path)
    
    # 恢复初始参数
    set_parameters(0, 0)
    X, Y = np.meshgrid(alpha_vals, beta_vals)
    Z = loss_grid.T
    
    plt.figure(figsize=(8, 6))
    
    if plot_type == '3d':
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, rstride=1, cstride=1,
                              edgecolor='none', antialiased=True)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_zlabel('Loss Value')
        plt.colorbar(surf, shrink=0.5)
    elif plot_type == '2d':
        plt.contourf(X, Y, Z, levels=20, cmap=cmap)
        plt.colorbar(label='Loss Value')
    else:
        raise ValueError("Invalid plot_type. Choose '2d' or '3d'")
    plt.savefig(save_name, dpi=500)
    
    return loss_grid

def display_images_horizontally(image_list, save_name):
    plt.clf()
    n = len(image_list)
    # 创建一行n列的子图，调整尺寸和间距
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    
    # 处理单张图的特殊情况（避免 axes 不是数组）
    if n == 1:
        axes = [axes]
    
    # 遍历显示每张图
    for i, ax in enumerate(axes):
        ax.imshow(image_list[i])  # 假设 image_list 中是 numpy 数组或 PIL 图像
        ax.axis('off')  # 关闭坐标轴
    
    # 自动调整子图间距（可根据需求调整参数）
    plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95)
    plt.savefig(save_name, dpi=500)

class ModelWrapper(torch.nn.Module):
    def __init__(self, image_model, text_features):
        super().__init__()
        self.image_model = image_model
        self.register_buffer("text_features", text_features)
    
    def forward(self, x):
        features = self.image_model(x)["adapted"]
        logit_scale = self.image_model.clip.logit_scale.exp()
        logits = logit_scale * features @ self.text_features.T
        return logits

class TemporaryGradEnable:
    def __init__(self, layers):
        self.layers = layers
        self.original_requires_grad = {}
        
    def __enter__(self):
        for layer in self.layers:
            for param in layer.parameters():
                self.original_requires_grad[param] = param.requires_grad
                param.requires_grad = True
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = self.original_requires_grad[param]

def find_value(dic, index):
    for item in dic:
        if item['index'] == index:
            return item

def find_class(dic, selected_classes):
    for item in dic:
        if item['selected_classes'] == selected_classes:
            return item

def test(model, test_loader, mode='adapted'):
    model.eval()
    correct = 0
    total = 0
    
    # 生成文本分类器
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_loader.dataset.classes]).cuda()
    text_features = clip_model.encode_text(text_inputs).float()
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            features = outputs[mode]
            
            logits_scale = clip_model.logit_scale.exp()
            logits = logits_scale * features @ text_features.T
            _, predicted = torch.max(logits, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"{mode.upper()} Accuracy: {acc:.2f}%")

    return acc

def moduleify(Gpt_output, init_net, unnormalize_fn):
    """
    Gpt_output: (N, D) tensor (N = batch_size, D = number of parameters)
    net_constructor: Function (should take no args/kwargs) that returns a randomly-initialized neural network
                     with the appropriate architecture
    unnormalize_fn: Function that takes a (N, D) tensor and "unnormalizes" it back to the original parameter space

    Returns: A length-N list of nn.Module instances, where the i-th nn.Module has the parameters from Gpt_output[i].
             If N = 1, then a single nn.Module is returned instead of a list.
    """
    Gpt_output = unnormalize_fn(Gpt_output)
    num_nets = Gpt_output.size(0)
    net_instance = deepcopy(init_net)
    target_state_dict = net_instance.state_dict()
    parameter_names = target_state_dict.keys()
    parameter_sizes = [v.size() for v in target_state_dict.values()]
    parameter_chunks = [v.numel() for v in target_state_dict.values()]

    parameters = torch.split(Gpt_output, parameter_chunks, dim=1)
    state_dicts = []
    for i in range(num_nets):
        # Build a state dict from the generated parameters:
        state_dict = {
            pname: param[i].reshape(size) for pname, param, size in \
                zip(parameter_names, parameters, parameter_sizes)
        }
        state_dicts.append(state_dict)
    return state_dicts
    

def create_thresholding_fn(thresholding, param_range):
    """
    Creates a function that thresholds after each diffusion sampling step.

    thresholding = "none": No thresholding.
    thresholding = "static": Clamp the sample to param_range.
    """

    if thresholding == "none":
        def denoised_fn(x):
            return x
    elif thresholding == "static":
        def denoised_fn(x):
            return torch.clamp(x, param_range[0], param_range[1])
    else:
        raise NotImplementedError

    return denoised_fn

def synth(
    diffusion,
    G,
    cond,         # G Condition
    w_shape,             # Shape of generate parameters
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process.
    """

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
        # print(f"Using thresholding={thresholding} with min_val={param_range[0]}, max_val={param_range[1]}")
    else:
        denoised_fn = None

    model_kwargs = {
        'cond':cond
    }

    shape = w_shape
    sample = diffusion.p_sample_loop(
        G,
        shape,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device='cuda',
        denoised_fn=denoised_fn,
        **p_sample_loop_kwargs
    )

    return sample

def synth_seq(
    diffusion,
    G,
    cond,         # G Condition
    w_shape,             # Shape of generate parameters
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process.
    """

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
        # print(f"Using thresholding={thresholding} with min_val={param_range[0]}, max_val={param_range[1]}")
    else:
        denoised_fn = None

    model_kwargs = {
        'cond':cond
    }

    shape = w_shape
    samples = []
    for sample in diffusion.p_sample_loop_progressive(
        G,
        shape,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device='cuda',
        denoised_fn=denoised_fn,
        **p_sample_loop_kwargs
    ):
        samples.append(sample['sample'])

    return samples


def run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict):
    """
    Computes the diffusion training loss for a batch of inputs.
    """
    p_data = batch_dict['p_data'].cuda()
    text_features = batch_dict['text'].cuda()
    t, vlb_weights = timestep_sampler.sample(p_data.shape[0], p_data.device)
    with torch.autocast('cuda', enabled=cfg.amp):
        model_kwargs = {
            'cond': text_features
        }
        losses = diffusion.training_losses(model, p_data, t, model_kwargs=model_kwargs)
    loss = (losses["loss"] * vlb_weights).mean()
    return loss, losses


def train_epoch(
    cfg, diffusion, model, model_module, ema, train_loader, timestep_sampler, optimizer, scaler, meter, epoch
):
    """
    Performs one epoch of G.pt training.
    """
    shuffle(train_loader, epoch)
    model.train()
    meter.reset()
    meter.iter_tic()
    epoch_iters = len(train_loader)
    for batch_ind, batch_dict in enumerate(train_loader):
        # print(batch_dict)
        lr = update_lr(cfg, optimizer, epoch + (batch_ind / epoch_iters))
        loss, loss_dict = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
        # print(loss)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if cfg.transformer.ema:
            accumulate(ema, model_module, cfg.train.ema_decay)
        loss_dict["loss"] = loss.view(1)
        loss_dict = {k: scaled_all_reduce(cfg, [v.mean()])[0].item() for k, v in loss_dict.items()}
        meter.iter_toc()
        meter.record_stats(loss_dict, lr)
        meter.log_iter_stats(epoch, batch_ind + 1)
        meter.iter_tic()
    meter.log_epoch_stats(epoch + 1)


def checkpoint_model(cfg, is_best_model, epoch, G_module, ema_module, optimizer, **save_dict):
    """
    Save a G.pt checkpoint.
    """
    periodic_checkpoint = epoch % cfg.train.checkpoint_freq == 0
    if is_best_model or periodic_checkpoint:
        if is_main_proc():
            base_path = f'{cfg.out_dir}/{cfg.exp_name}/checkpoints'
            save_dict.update({
                'G': G_module.state_dict(),
                'optim': optimizer.state_dict()
            })
            if cfg.transformer.ema:
                save_dict.update({'G_ema': ema_module.state_dict()})
            if is_best_model:
                torch.save(save_dict, f'{base_path}/best.pt')
            if periodic_checkpoint:
                torch.save(save_dict, f'{base_path}/{epoch:04}.pt')
        synchronize()


@torch.inference_mode()
def test_epoch(cfg, diffusion, model, test_loader, timestep_sampler, meter, epoch):
    """
    Evaluate G.pt on test set (unseen) neural networks.
    """
    if (epoch + 1) % cfg.test.freq == 0:
        model.eval()
        meter.reset()
        for batch_ind, batch_dict in enumerate(test_loader):
            loss, loss_dict = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
            loss_dict["loss"] = loss.view(1)
            loss_dict = {
                k: scaled_all_reduce(cfg, [v.mean()])[0].item() for k, v in loss_dict.items()
            }
            meter.record_stats(loss_dict)
        meter.log_epoch_stats(epoch + 1)

def eval_generate_model(
    diffusion,
    G,
    cond,
    selected_classes_set,
    denormalize,
    w_shape
):
    sample = synth(
        diffusion=diffusion,
        G=G,
        cond=cond,
        w_shape=w_shape
    )
    print('Now Sample:', sample.shape)
    init_net = CLIPAdapter(hidden_dim=16)
    g_nets = []
    state_dicts = moduleify(sample, init_net.resnet.fc, denormalize)
    for state_dict in state_dicts:
        now_net = deepcopy(init_net)
        now_net.resnet.fc.load_state_dict(state_dict)
        g_nets.append(now_net)

    accs = []

    for i in range(len(selected_classes_set)):
        test_dataset = Caltech256Subset(root='./data_gen/caltech256/data', split='test', class_names=selected_classes_set[i])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
        net = g_nets[i]
        acc = test(net.cuda(), test_loader, mode='adapted')
        accs.append(acc)
    return sum(accs)/len(accs)

def eval_denoise_model(
    diffusion,
    G,
    cond,
    selected_classes_set,
    denormalize,
    w_shape,
    steps=[1000]
):
    sample = synth_seq(
        diffusion=diffusion,
        G=G,
        cond=cond,
        w_shape=w_shape
    )
    init_sample = torch.stack(sample, dim=0)
    ret_acc = {}
    ret_nets = {}
    ret_ds = {}
    for step in steps:
        sample = init_sample[step]
        init_net = CLIPAdapter(clip_model=clip_model ,hidden_dim=16)
        g_nets = []
        state_dicts = moduleify(sample, init_net.resnet.fc, denormalize)
        for state_dict in state_dicts:
            now_net = deepcopy(init_net)
            now_net.resnet.fc.load_state_dict(state_dict)
            g_nets.append(now_net)

        accs = []
        ds = []

        for i in range(len(selected_classes_set)):
            test_dataset = Caltech256Subset(root='./data_gen/caltech256/data', split='test', class_names=selected_classes_set[i])
            ds.append(test_dataset)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
            net = g_nets[i]
            acc = test(net.cuda(), test_loader, mode='adapted')
            accs.append(acc)

        # del g_nets
        # del ds

        ret_nets[str(step)] = g_nets
        ret_acc[str(step)] = accs
        ret_ds[str(step)] = ds
        
    return ret_acc, ret_nets, ret_ds

def cam_vis(clip_adapter:CLIPAdapter, dataset:CIFAR100Subset, device, name, data_index):

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean
    
    clip_adapter.eval()

    classes = dataset.classes
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    text_features = clip_adapter.encode_text(text_inputs)

    wrapped_model = ModelWrapper(clip_adapter, text_features).to(device)
    # print(clip_adapter)
    target_layer = clip_adapter.resnet.layer2[-1].conv2
    target_layers = [target_layer]

    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    image, label = dataset[data_index]
    image = image.unsqueeze(0).to(device)
    label = torch.tensor(label).unsqueeze(0).to(device)

    with TemporaryGradEnable(target_layers):
        targets = [ClassifierOutputTarget(label.item())]
        grayscale_cam = cam(input_tensor=image, targets=targets)

    rgb_img = denormalize(image).cpu().squeeze().permute(1, 2, 0).numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    return cam_image

def vis_loss(g_nets, ds, x0_net_paths, device):

    classes = ds['900'][0].classes
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    text_features = g_nets['900'][0].encode_text(text_inputs)

    dataset = ds['900'][0]
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    indices = indices[:int(len(dataset)*0.1)]
    dataset = torch.utils.data.Subset(dataset,indices=indices)
    dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=8)

    init_net_sd = torch.load('./data_gen/cifar100/checkpoints/base.pt', map_location='cpu')
    init_net = CLIPAdapter(clip_model=clip_model, hidden_dim=16).to(device)
    init_net.resnet.fc.load_state_dict(init_net_sd['clip_adapter'])

    a_net_sd = torch.load(x0_net_paths[0], map_location='cpu')
    b_net_sd = torch.load('./data_gen/cifar100/checkpoints/10/task.pt', map_location='cpu') #some other net

    a_net = CLIPAdapter(clip_model=clip_model, hidden_dim=16).to(device)
    b_net = CLIPAdapter(clip_model=clip_model, hidden_dim=16).to(device)
    a_net.resnet.fc.load_state_dict(a_net_sd['clip_adapter'])
    b_net.resnet.fc.load_state_dict(b_net_sd['clip_adapter'])

    init_net_wrap = ModelWrapper(image_model=init_net, text_features=text_features)
    a_net_wrap = ModelWrapper(image_model=a_net, text_features=text_features)
    b_net_wrap = ModelWrapper(image_model=b_net, text_features=text_features)

    initial_params = [p.clone() for p in init_net_wrap.parameters()]
    delta1 = [p - init_p for p, init_p in zip(a_net_wrap.parameters(), initial_params)]
    delta2 = [p - init_p for p, init_p in zip(b_net_wrap.parameters(), initial_params)]

    res = visualize_2d_loss_landscape(
        model=a_net_wrap,
        initial_params=initial_params,
        delta1=delta1,
        delta2=delta2,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=dataloader,
        plot_type='3d',
        cmap='coolwarm',
        elevation=35,
        azimuth=20,
        save_name='loss_landscope_3d.png'
    )

    return res


def vis(cfg):
    seed = setup_env(cfg)
    train_dataset = PDataset(
        config_path='./data_gen/caltech256/train.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )
    test_dataset = PDataset(
        config_path='./data_gen/caltech256/val.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )
    # Construct the model and optimizer
    model = Gpt(
        parameter_sizes=train_dataset.parameter_sizes,
        parameter_names=train_dataset.parameter_names,
        predict_xstart=cfg.transformer.predict_xstart,
        absolute_loss_conditioning=cfg.transformer.absolute_loss_conditioning,
        chunk_size=cfg.transformer.chunk_size,
        split_policy=cfg.transformer.split_policy,
        max_freq_log2=cfg.transformer.max_freq_log2,
        num_frequencies=cfg.transformer.num_frequencies,
        n_embd=cfg.transformer.n_embd,
        encoder_depth=cfg.transformer.encoder_depth,
        decoder_depth=cfg.transformer.decoder_depth,
        n_layer=cfg.transformer.n_layer,
        n_head=cfg.transformer.n_head,
        attn_pdrop=cfg.transformer.dropout_prob,
        resid_pdrop=cfg.transformer.dropout_prob,
        embd_pdrop=cfg.transformer.dropout_prob
    )
    diffusion = create_diffusion(
        learn_sigma=False, predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule='linear', steps=1000
    )
    cur_device = torch.cuda.current_device()
    print(cur_device)
    model = model.cuda(device=cur_device)

    G_path = '/data/bowen/Text2Weight/results/caltech256_classifier/checkpoints/best.pt'
    G_ckpt = torch.load(G_path, map_location=f"cuda:{cur_device}")
    G_ckpt = G_ckpt['G']

    model.load_state_dict(G_ckpt)
    model = model.cuda(device=cur_device)
    model.eval()

    conds = []
    selected_classes_set = []
    p_datas = []
    x0_net_paths = []
    count = 0
    with open('./data_gen/caltech256/selected_point.json','r') as f:
        data_point = json.load(f)
    for item in test_dataset:
        if item['index'] != data_point['index']:continue
        cond = item['text']
        index = item['index']
        p_data = item['p_data']
        p_datas.append(p_data)
        conds.append(cond)
        info = find_value(test_dataset.data, index)
        x0_net_paths.append(info['path'])
        selected_classes_set.append(info['selected_classes'])
        count += 1
        if count == 1:break
    conds = torch.stack(conds, dim=0)
    conds = conds.cuda(device=cur_device)
    p_datas = torch.stack(p_datas, dim=0)
    print('cond:',conds.shape)
    print('p_datas:',p_datas.shape)

    # steps = [900,910,920,930,940,950,960,970,980,990,999]
    steps = [999]
    # steps = [1, 300, 500, 700, 900, 930, 960, 999]

    # file_name = 'vis_data.pt'
    # if os.path.exists(file_name):
    #     data = torch.load(file_name)
    #     result = data['result']
    #     g_nets = data['g_nets']
    #     ds = data['ds']
    # else:
    result, g_nets, ds = eval_denoise_model(
        diffusion=diffusion,
        G=model,
        cond=conds,
        selected_classes_set=selected_classes_set,
        denormalize=test_dataset.unnormalize,
        w_shape=p_datas.shape,
        steps=steps
    )
    save_dict = {
        'accuracy':0.0,
        'clip_adapter':g_nets['999'][0].resnet.fc.state_dict()
    }
    torch.save(save_dict, './caltech_fusion.pt')

    # data = {}
    # data['result'] = result
    # data['g_nets'] = g_nets
    # data['ds'] = ds
        # torch.save(data, file_name)

    # res = vis_loss(g_nets, ds, x0_net_paths, device='cuda')
    # res.savefig('./loss_land.png',dpi=500)

    # print(result)
    # index_range = [i for i in range(len(ds['900'][0]))]
    # random.shuffle(index_range)
    # index_range = index_range[:100]
    # for i in index_range:
    #     img_list = []
    #     for step in steps:
    #         print(f'Generate Step:{step}, Index:{i}')
    #         img = cam_vis(g_nets[str(step)][0],ds['900'][0],device='cuda',name=f'./vis/{step}-{i}.png',data_index=i)
    #         img_list.append(img)

    #     display_images_horizontally(img_list, save_name=f'./vis/{i}.png')


    # mean_acc = eval_generate_model(
    #     diffusion=diffusion,
    #     G=model,
    #     cond=conds,
    #     selected_classes_set=selected_classes_set,
    #     denormalize=test_dataset.unnormalize,
    #     w_shape=p_datas.shape,
    # )


@hydra.main(config_path="configs/train", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    vis(cfg)

if __name__ == '__main__':
    main()
