#!/usr/bin/env python3
"""
Script for training and evaluating G.pt models.
"""
try:
    import isaacgym
except ImportError:
    print("WARNING: Isaac Gym not imported")
import clip
import sys
import tqdm
import multiprocessing
import os
import hydra
import json
import omegaconf
import random
from copy import deepcopy
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from T2W.diffusion import create_diffusion
from T2W.diffusion.timestep_sampler import UniformSampler

from data_gen.cifar100.p_dataset import PDataset
from T2W.distributed import scaled_all_reduce
from T2W.models.transformer import Gpt
from T2W.meters import TrainMeter, TestMeter
from T2W.utils import setup_env, construct_loader, shuffle, update_lr, spread_losses, accumulate, requires_grad
from T2W.distributed import get_rank, get_world_size, is_main_proc, synchronize
# from Gpt.vis import VisMonitor
# from Gpt.tasks import get
from data_gen.caltech256.generate_dataset import CLIPAdapter, Caltech256Subset

clip_model,_ = clip.load("ViT-B/32", device='cuda')

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def find_value(dic, index):
    for item in dic:
        if item['index'] == index:
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
    pred = losses['pred']
    loss = (losses["loss"] * vlb_weights).mean()
    return loss, losses, pred


def train_epoch(
    cfg, diffusion, model, model_module, ema, train_loader, timestep_sampler, optimizer, scaler, meter, epoch, d_optim, discriminator
):
    d_loss = []
    diff_loss = []
    adv_loss = []
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
        loss, loss_dict, x_start = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
        # print(loss)
        d_fake = discriminator(x_start)
        loss_g_adv = torch.nn.functional.binary_cross_entropy_with_logits(
            d_fake, torch.ones_like(d_fake)
        )
        diff_loss.append(loss.item())
        adv_loss.append(loss_g_adv.item())
        loss = loss + loss_g_adv * 0
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
        

    for batch_ind, batch_dict in enumerate(train_loader):
        with torch.no_grad():
            _, _, x_start = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
        real_data = batch_dict['p_data'].cuda()
        fake_data = x_start.detach()
        # 计算判别器损失
        d_optim.zero_grad()
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)
        
        loss_d_real = torch.nn.functional.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        loss_d_fake = torch.nn.functional.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake)
        )
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        d_loss.append(loss_d.item())
        
        loss_d.backward()
        d_optim.step()

    res = {
        'adv_loss':adv_loss,
        'diff_loss':diff_loss,
        'd_loss':d_loss
    }
    logs = {}
    for key in res.keys():
        logs[key] = sum(res[key])/len(res[key])
    print(f'In Epoch {epoch+1}')
    print(logs)
    sys.stdout.flush()

    # meter.log_epoch_stats(epoch + 1)
    return res


def checkpoint_model(cfg, is_best_model, epoch, G_module, ema_module, optimizer, **save_dict):
    """
    Save a G.pt checkpoint.
    """
    periodic_checkpoint = epoch % cfg.train.checkpoint_freq == 0
    if is_best_model or periodic_checkpoint:
        print('Save Model')
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
        diff_loss = []
        for batch_ind, batch_dict in enumerate(test_loader):
            loss, loss_dict, _ = run_diffusion_vlb(cfg, diffusion, model, timestep_sampler, batch_dict)
            loss_dict["loss"] = loss.view(1)
            loss_dict = {
                k: scaled_all_reduce(cfg, [v.mean()])[0].item() for k, v in loss_dict.items()
            }
            diff_loss.append(loss.item())

        print(f'Test Loss Is:{sum(diff_loss)/len(diff_loss)}')
        


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

def train(cfg):
    """Performs the full training loop."""

    # Set up the environment
    seed = setup_env(cfg)

    # Instantiate visualization objects (they will be fully-initialized later on)
    # vis_monitor = VisMonitor(
    #     cfg.dataset.name,
    #     None,
    #     None,
    #     net_mb_size=cfg.vis.net_mb_size_per_gpu,
    #     vis_recursion=cfg.vis.recursive_probe,
    #     vis_period=cfg.vis.freq,
    #     delay_test_fn=True,
    #     dvo_steps=cfg.vis.dvo_steps,
    #     prompt_start_coeff=cfg.vis.prompt_start_coeff,
    #     thresholding=cfg.sampling.thresholding,
    #     param_range=None
    # )

    train_dataset = PDataset(
        config_path='./data_gen/caltech256/train-natural.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )

    test_dataset = PDataset(
        config_path='./data_gen/caltech256/val-natural.json',
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff
    )

    # Construct data loaders
    train_loader = construct_loader(
        train_dataset, cfg.train.mb_size, cfg.num_gpus,
        shuffle=True, drop_last=True, num_workers=cfg.dataset.num_workers
    )
    test_loader = construct_loader(
        test_dataset, cfg.test.mb_size, cfg.num_gpus,
        shuffle=False, drop_last=False, num_workers=cfg.dataset.num_workers
    )

    # Construct meters
    train_meter = TrainMeter(len(train_loader), cfg.train.num_ep)
    test_meter = TestMeter(len(test_loader), cfg.train.num_ep)

    total = 0
    for item in train_dataset.parameter_sizes:
        total += sum(item)
    input_dim = total
    discriminator = Discriminator(input_dim).cuda()
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=0.01, 
        betas=(0.5, 0.999)
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

    # Create an exponential moving average (EMA) of G.pt
    if cfg.transformer.ema:
        ema = deepcopy(model)
        requires_grad(ema, False)
    else:
        ema = None

    # Diffusion objects
    diffusion = create_diffusion(
        learn_sigma=False, predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule='linear', steps=1000
    )
    timestep_sampler = UniformSampler(diffusion)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    # Transfer model to GPU
    cur_device = torch.cuda.current_device()
    # print(cur_device)
    model = model.cuda(device=cur_device)
    if cfg.transformer.ema:
        ema = ema.cuda(device=cur_device)

    # Use DDP for multi-gpu training
    if (not cfg.test_only) and (cfg.num_gpus > 1):
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            static_graph=True
        )
        model.configure_optimizers = model.module.configure_optimizers
        module = model.module
    else:
        module = model

    # Initialize the EMA model with an exact copy of weights
    if cfg.transformer.ema and cfg.resume_path is None:
        accumulate(ema, module, 0)

    # Construct the optimizer
    optimizer = model.configure_optimizers(
        lr=cfg.train.base_lr,
        wd=cfg.train.wd,
        betas=(0.9, cfg.train.beta2)
    )

    # Resume from checkpoint
    if cfg.resume_path is not None:
        resume_checkpoint = find_model(cfg.resume_path)
        module.load_state_dict(resume_checkpoint['G'])
        if cfg.transformer.ema:
            ema.load_state_dict(resume_checkpoint['G_ema'])
        if not cfg.test_only:
            optimizer.load_state_dict(resume_checkpoint['optim'])
        try:
            start_epoch = int(os.path.basename(cfg.resume_path).split('.')[0])
        except ValueError:
            start_epoch = 0
        print(f'Resumed G.pt from checkpoint {cfg.resume_path}, using start_epoch={start_epoch}')
    else:
        start_epoch = 0
        print('Training from scratch')

    model2test = ema if cfg.transformer.ema and cfg.test.use_ema else model

    print('Beginning training...')
    best_test_acc = 0.
    loss_record = []
    for epoch in range(start_epoch, cfg.train.num_ep):
        res = train_epoch(
            cfg, diffusion, model, module, ema, train_loader, timestep_sampler, optimizer, scaler,
            train_meter, epoch, d_optim=d_optimizer, discriminator=discriminator
        )
        loss_record.append(res)
        if epoch % 50 == 0 or epoch + 1 == cfg.train.num_ep:
            with open(cfg.exp_name+'.json', 'w') as f:
                json.dump(loss_record,f)
        test_epoch(cfg, diffusion, model2test, test_loader, timestep_sampler, test_meter, epoch)

        if (epoch + 1) % 50 == 0:
            batch = next(iter(test_loader))
            cond = batch['text'].cuda()
            indexes = batch['index'].cpu().tolist()
            selected_classes_set = []
            for index in indexes:
                item = find_value(test_dataset.data, index)
                selected_classes_set.append(item['selected_classes'])

            mean_acc = eval_generate_model(
                diffusion=diffusion,
                G=model2test,
                cond=cond,
                selected_classes_set=selected_classes_set,
                denormalize=test_dataset.unnormalize,
                w_shape=batch['p_data'].shape,
            )
            mean_acc = eval_generate_model(
                diffusion=diffusion,
                G=model2test,
                cond=cond,
                selected_classes_set=selected_classes_set,
                denormalize=test_dataset.unnormalize,
                w_shape=batch['p_data'].shape,
            )
            
        new_best_model = True
        checkpoint_model(cfg, new_best_model, epoch + 1, module, ema, optimizer)


def single_proc_train(local_rank, port, world_size, cfg):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:{}".format(port),
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    train(cfg)
    torch.distributed.destroy_process_group()
    exit()


@hydra.main(config_path="configs/train", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):

    # Multi-gpu training
    if cfg.num_gpus > 1:
        # Select a port for proc group init randomly
        port_range = [10000, 65000]
        port = random.randint(port_range[0], port_range[1])
        # Start a process per-GPU:
        torch.multiprocessing.start_processes(
            single_proc_train,
            args=(port, cfg.num_gpus, cfg),
            nprocs=cfg.num_gpus,
            start_method="spawn"
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
