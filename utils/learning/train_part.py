import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy
import cv2

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
# from utils.model.varnet import VarNet
from utils.model.feature_varnet import FeatureVarNet_sh_w as VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, using_noise_mask=False):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, _, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        if using_noise_mask:
            target, output = apply_mask_to_target_and_reconstruction(target, output, modality=args.modality)

        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
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
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            if using_noise_mask:
                target, output = apply_mask_to_target_and_reconstruction(target, output, modality=args.modality)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


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
                   pools=args.pools)
    model.to(device=device)

    # 모델 파라미터 수 출력
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {num_params}")

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, modality=args.modality)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, shuffle=False, modality=args.modality)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, args.using_noise_mask)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, args.using_noise_mask)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
