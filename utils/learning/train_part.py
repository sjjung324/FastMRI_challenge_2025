import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy
import cv2
import random
# from torch.amp import GradScaler, autocast
import os

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, L1Loss
# from utils.model.varnet import VarNet
from utils.model.feature_varnet import FeatureVarNet_sh_w as VarNet
from utils.data.augmentation import augment_kspace

from contextlib import contextmanager
import os
import pathlib

@contextmanager
def _patch_posixpath_on_windows():
    if os.name == 'nt' and hasattr(pathlib, 'PosixPath'):
        _orig = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
            yield
        finally:
            pathlib.PosixPath = _orig
    else:
        yield

def train_epoch(args, epoch, model, data_loader, optimizer, using_noise_mask=False, using_augmentation=False, scaler=None, augmented_loader=None):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    # Exponential ramp-up schedule for augmentation probability
    p_max = getattr(args, 'aug_p_max', 0.55)
    c = getattr(args, 'aug_curve', 3)
    T = getattr(args, 'num_aug_epochs', 0) + 1
    t = epoch - args.num_epochs
    p_aug = p_max * (1 - np.exp(-c * t / T)) / (1 - np.exp(-c))

    alpha = args.alpha
    L1_loss = L1Loss().to(device=device)
    SSIM_loss = SSIMLoss().to(device=device)

    if using_augmentation and epoch > args.num_epochs:
        data_loader = augmented_loader

    for iter, data in enumerate(data_loader):
        mask, kspace, _, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        # augment_config = {
        #     'flip': random.choice([True, False]),
        #     'flip_horizontal': True,
        #     'translate': random.choice([True, False]),
        #     # 'translate': False,
        #     'translate_max': 3,
        #     'affine': True,
        #     'affine_rot': 3.0,
        #     'affine_scale': 0.03,
        #     'affine_shear': 0.0
        # }
        # # Apply augmentation with probability p_aug
        # if using_augmentation and random.random() < p_aug:
        #     kspace, target = augment_kspace(kspace, target, augment_config)

        # with autocast(dtype=torch.bfloat16, device_type='cuda'):
        #     output = model(kspace, mask)
        output = model(kspace, mask)

        if using_noise_mask:
            target, output = apply_mask_to_target_and_reconstruction(target, output, modality=args.modality)

        # with autocast(dtype=torch.bfloat16, device_type='cuda'):
        #     loss = alpha * SSIM_loss(output, target, maximum) + (1 - alpha) * L1_loss(output, target, maximum)
        loss = alpha * SSIM_loss(output, target, maximum) + (1 - alpha) * L1_loss(output, target, maximum) * 10**4

        
        accumulation_steps = args.accumulation_steps
        loss = loss / accumulation_steps
        loss.backward()
        if (iter + 1) % accumulation_steps == 0 or (iter + 1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item() * accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{T - 1 + args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, using_noise_mask=False):
    model.eval()
    reconstructions = defaultdict(dict)
    maximums = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, target, maximum, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = model(kspace, mask)

       
            if using_noise_mask:
                target, output = apply_mask_to_target_and_reconstruction(target, output, modality=args.modality)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
                maximums[fnames[i]][int(slices[i])] = maximum[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    if args.data_path_val_additional is None:
        for fname in maximums:
            maximums[fname] = np.stack(
                [out for _, out in sorted(maximums[fname].items())]
            )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname], maxval=maximums[fname]) for fname in reconstructions])
    else:
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname], maxval=maximum.cpu().numpy()) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def validate_on_gpu(args, model, data_loader, using_noise_mask=False):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    start = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    SSIM_loss = SSIMLoss().to(device=device)

    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, target, maximum, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = model(kspace, mask)
            if using_noise_mask:
                target, output = apply_mask_to_target_and_reconstruction(target, output, modality=args.modality)

            loss = SSIM_loss(output, target, maximum)
            total_loss += loss.item()

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    return total_loss, len_loader, reconstructions, targets, None, time.perf_counter() - start
    

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def compute_mask_from_target(target, modality='all'):
    """
    Given a target (numpy array or torch tensor) and part name ('knee_test' or 'brain_test'),
    compute and return the mask using the specified logic.
    """
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target
    mask = np.zeros(target_np.shape, dtype=np.uint8)
    if modality == 'knee' or modality == 'all':
        mask[target_np > 2e-5] = 1
    elif modality == 'brain':
        mask[target_np > 5e-5] = 1
    else:
        raise ValueError(f"Invalid modality '{modality}'. Choose from 'all', 'brain', or 'knee'.")
    
    kernel = np.ones((3, 3), np.uint8)
    # If target is 3D (batch), apply per slice
    if mask.ndim == 3:
        for i in range(mask.shape[0]):
            mask[i] = cv2.erode(mask[i], kernel, iterations=1)
            mask[i] = cv2.dilate(mask[i], kernel, iterations=15)
            mask[i] = cv2.erode(mask[i], kernel, iterations=14)
    else:
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=15)
        mask = cv2.erode(mask, kernel, iterations=14)

    mask = (torch.from_numpy(mask).to(target)).type(torch.float)
    return mask

def apply_mask_to_target_and_reconstruction(target, reconstruction, modality='all'):
    """
    Apply the mask to both target and reconstruction.
    """

    mask = compute_mask_from_target(target, modality)
    masked_target = target * mask
    masked_reconstruction = reconstruction * mask
    return masked_target, masked_reconstruction

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())


    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   sens_pools=args.sens_pools,
                   pools=args.pools,
                   using_memory_efficient= not args.no_using_memory_efficient,
                   using_cpu_memory=not args.no_using_cpu_memory
                   )

    base_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), base_lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, threshold=3e-4, threshold_mode='abs', cooldown=1, min_lr=4e-5)
    
    val_loss_log = np.empty((0, 2))
    train_loss_log = np.empty((0, 2))
    lr_log = np.empty((0, 2))

    if args.pretrained_model_path is not None:
        with _patch_posixpath_on_windows():
            model_ckpt = torch.load(args.pretrained_model_path, map_location='cpu', weights_only=False)
            if 'model' in model_ckpt:
                model.load_state_dict(model_ckpt['model'])
                print("Pretrained Model Loaded")

                if "optimizer" in model_ckpt:
                    optimizer.load_state_dict(model_ckpt["optimizer"])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)
                    print("Optimizer state loaded.")
            
            else:
                model.load_state_dict(model_ckpt)

    if args.resume:
        if os.path.exists(args.exp_dir / 'model.pt'):
            model_ckpt = torch.load(args.exp_dir / 'model.pt', map_location=device, weights_only=False)
            if 'model' in model_ckpt:
                model.load_state_dict(model_ckpt['model'])
            else:
                model.load_state_dict(model_ckpt)
            print(f"Model loaded from {args.exp_dir / 'model.pt'}")

        if os.path.exists(args.val_loss_dir / 'lr_log.npy'):
            lr_log = np.load(args.val_loss_dir / 'lr_log.npy')
            print(f"lr_log loaded from {args.val_loss_dir / 'lr_log.npy'}")
        if os.path.exists(args.val_loss_dir / 'train_loss_log.npy'):
            train_loss_log = np.load(args.val_loss_dir / 'train_loss_log.npy')
            print(f"train_loss_log loaded from {args.val_loss_dir / 'train_loss_log.npy'}")
        if os.path.exists(args.val_loss_dir / 'val_loss_log.npy'):
            val_loss_log = np.load(args.val_loss_dir / 'val_loss_log.npy')
            print(f"val_loss_log loaded from {args.val_loss_dir / 'val_loss_log.npy'}")

    model.to(device=device)

    # 모델 파라미터 수 출력
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {num_params}")

    # 모델 용량 출력
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # in MB
    print(f"Model size: {model_size:.2f} MB")

    best_val_loss = 1.
    start_epoch = 1
    start_epoch = 1

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, additional_root=args.data_path_train_additional)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, shuffle=False, additional_root=args.data_path_val_additional)
    augmented_train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, kspace_augment=True, additional_root=args.data_path_train_additional)

    num_epochs = args.num_epochs + args.num_aug_epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, args.using_noise_mask, using_augmentation=(epoch > args.num_epochs), augmented_loader=augmented_train_loader)
        if args.validate_on_gpu:
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_on_gpu(args, model, val_loader, args.using_noise_mask)
        else:
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, args.using_noise_mask)
        
        scheduler.step(val_loss / num_subjects)

        lr_log = np.append(lr_log, np.array([[epoch, optimizer.param_groups[0]['lr']]]), axis=0)
        train_loss_log = np.append(train_loss_log, np.array([[epoch, train_loss]]), axis=0)
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss / num_subjects]]), axis=0)

        lr_log_file_path = os.path.join(args.val_loss_dir, "lr_log")
        np.save(lr_log_file_path, lr_log)
        train_loss_file_path = os.path.join(args.val_loss_dir, "train_loss_log")
        np.save(train_loss_file_path, train_loss_log)
        val_loss_file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(val_loss_file_path, val_loss_log)
        print(f"loss file saved! {val_loss_file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            f'lr = {optimizer.param_groups[0]["lr"]:.6f}'
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
