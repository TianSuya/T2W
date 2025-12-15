#!/usr/bin/env python3
"""
Evaluation script for T2W (Text-to-Weight) model.
Part of the code was modified from repository https://github.com/wpeebles/G.pt.
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
from T2W.diffusion import create_diffusion
from T2W.diffusion.timestep_sampler import UniformSampler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from data_gen.cifar100.p_dataset import PDataset
from T2W.distributed import scaled_all_reduce
from T2W.models.transformer import T2W as T2WModel
import joblib
from T2W.meters import TrainMeter, TestMeter
from T2W.utils import setup_env, construct_loader, shuffle, update_lr, spread_losses, accumulate, requires_grad
from T2W.distributed import get_rank, get_world_size, is_main_proc, synchronize
from data_gen.cifar100.generate_dataset import CLIPAdapter, CIFAR100Subset
import numpy as np
from matplotlib.colors import Normalize
import json

# Configuration: modify this path according to your setup
DATA_ROOT = "./data/cifar100"
CHECKPOINT_PATH = "./results/cifar100_classifier/checkpoints/best.pt"

clip_model, _ = clip.load("ViT-B/32", device='cuda')


def find_model(model_path):
    """Load T2W model checkpoint."""
    assert os.path.isfile(model_path), f'Could not find T2W checkpoint at {model_path}'
    return torch.load(model_path, map_location=lambda storage, loc: storage)


def find_value(dic, index):
    for item in dic:
        if item['index'] == index:
            return item


def test(model, test_loader, mode='adapted'):
    model.eval()
    correct = 0
    total = 0
    
    # Generate text classifier
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


def moduleify(t2w_output, init_net, unnormalize_fn):
    """
    Convert T2W output to neural network modules.
    
    Args:
        t2w_output: (N, D) tensor (N = batch_size, D = number of parameters)
        init_net: Template neural network with the appropriate architecture
        unnormalize_fn: Function that takes a (N, D) tensor and "unnormalizes" it back to the original parameter space

    Returns: 
        A list of state_dicts, where the i-th state_dict has the parameters from t2w_output[i].
    """
    t2w_output = unnormalize_fn(t2w_output)
    num_nets = t2w_output.size(0)
    net_instance = deepcopy(init_net)
    target_state_dict = net_instance.state_dict()
    parameter_names = target_state_dict.keys()
    parameter_sizes = [v.size() for v in target_state_dict.values()]
    parameter_chunks = [v.numel() for v in target_state_dict.values()]

    parameters = torch.split(t2w_output, parameter_chunks, dim=1)
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
    cond,                # T2W Condition
    w_shape,             # Shape of generate parameters
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from T2W via the reverse diffusion process.
    """

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
    else:
        denoised_fn = None

    model_kwargs = {
        'cond': cond
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
    cond,                # T2W Condition
    w_shape,             # Shape of generate parameters
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from T2W via the reverse diffusion process with intermediate results.
    """

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
    else:
        denoised_fn = None

    model_kwargs = {
        'cond': cond
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
    Performs one epoch of T2W training.
    """
    shuffle(train_loader, epoch)
    model.train()
    meter.reset()
    meter.iter_tic()
    epoch_iters = len(train_loader)
    for batch_ind, batch_dict in enumerate(train_loader):
        lr = update_lr(cfg, optimizer, epoch + (batch_ind / epoch_iters))
        loss, loss_dict = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
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
    Save a T2W checkpoint.
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
    Evaluate T2W on test set (unseen) neural networks.
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
    w_shape,
    w=None,
    loss_fn=None,
):
    """
    Evaluate the generated model weights on downstream classification tasks.
    """
    sample = synth(
        diffusion=diffusion,
        G=G,
        cond=cond,
        w_shape=w_shape
    )
    loss = loss_fn(w, sample)
    print('Now Sample:', sample.shape)
    init_net = CLIPAdapter(clip_model=clip_model, hidden_dim=16)
    g_nets = []
    state_dicts = moduleify(sample, init_net.resnet.fc, denormalize)
    for state_dict in state_dicts:
        now_net = deepcopy(init_net)
        now_net.resnet.fc.load_state_dict(state_dict)
        g_nets.append(now_net)

    accs = []
    res = {}

    for i in range(len(selected_classes_set)):
        test_dataset = CIFAR100Subset(DATA_ROOT, train=False, classes=selected_classes_set[i])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
        net = g_nets[i]
        acc = test(net.cuda(), test_loader, mode='adapted')
        accs.append(acc)
    res['acc'] = sum(accs) / len(accs)
    res['loss'] = loss.item()
    return res


def eval_denoise_model(
    diffusion,
    G,
    cond,
    selected_classes_set,
    denormalize,
    w_shape,
    steps=[1000]
):
    """
    Evaluate the denoising process at different timesteps.
    """
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
        init_net = CLIPAdapter(clip_model=clip_model, hidden_dim=16)
        g_nets = []
        state_dicts = moduleify(sample, init_net.resnet.fc, denormalize)
        for state_dict in state_dicts:
            now_net = deepcopy(init_net)
            now_net.resnet.fc.load_state_dict(state_dict)
            g_nets.append(now_net)

        accs = []
        ds = []

        for i in range(len(selected_classes_set)):
            test_dataset = CIFAR100Subset(DATA_ROOT, train=False, classes=selected_classes_set[i])
            ds.append(test_dataset)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16)
            # Load base model for comparison
            base_net_sd = torch.load('./data_gen/cifar100/checkpoints/base.pt', map_location='cpu')
            net = deepcopy(init_net)
            net.resnet.fc.load_state_dict(base_net_sd['clip_adapter'])
            acc = test(net.cuda(), test_loader, mode='adapted')
            accs.append(acc)
        ret_nets[str(step)] = g_nets
        ret_acc[str(step)] = sum(accs) / len(accs)
        ret_ds[str(step)] = ds
        
    return ret_acc, ret_nets, ret_ds


def eval_model(cfg):
    """
    Main evaluation function for T2W model.
    """
    seed = setup_env(cfg)
    
    # Load datasets
    p_train_dataset = PDataset(
        config_path='./data_gen/cifar100/train.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )
    p_test_dataset = PDataset(
        config_path='./data_gen/cifar100/val.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )
    p_test_loader = DataLoader(dataset=p_train_dataset, batch_size=32, shuffle=False, num_workers=16)
    
    # Construct the T2W model
    model = T2WModel(
        parameter_sizes=p_train_dataset.parameter_sizes,
        parameter_names=p_train_dataset.parameter_names,
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
    
    # Create diffusion
    diffusion = create_diffusion(
        learn_sigma=False, predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule='linear', steps=1000
    )
    cur_device = torch.cuda.current_device()
    print(f"Using device: cuda:{cur_device}")
    model = model.cuda(device=cur_device)

    # Load T2W checkpoint
    t2w_ckpt = find_model(CHECKPOINT_PATH)
    t2w_ckpt = t2w_ckpt['G']

    model.load_state_dict(t2w_ckpt)
    model = model.cuda(device=cur_device)
    model.eval()

    # Evaluation loop
    selected_classes_set = []
    accuracys = []
    count = 0
    for batch in p_test_loader:
        cond = batch['text'].cuda()
        indexes = batch['index'].cpu().tolist()
        selected_classes_set = []
        for index in indexes:
            item = find_value(p_test_loader.dataset.data, index)
            selected_classes_set.append(item['selected_classes'])

        mean_acc = eval_generate_model(
            diffusion=diffusion,
            G=model,
            cond=cond,
            selected_classes_set=selected_classes_set,
            denormalize=p_test_dataset.unnormalize,
            w_shape=batch['p_data'].shape,
            w=batch['p_data'].cuda(),
            loss_fn=torch.nn.MSELoss()
        )
        accuracys.append(mean_acc)
        with open('./cifar100_eval_results.json', 'w') as f:
            json.dump(accuracys, f)
        count += 1
        if count >= 4:
            break


@hydra.main(config_path="configs/train", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    eval_model(cfg)


if __name__ == '__main__':
    main()
