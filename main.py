# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
#import hostlist
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dataset import LNENDataset
from model import BarlowTwins
from lars_opimizers import LARS
##########################
#  TRAIN
##########################
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='1024-512-256-128', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=1000, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=1000, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--checkpoint-dir', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/dev_nshape/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--list-dir',  default='TrainTumorNormal.txt', type=str, metavar='C',
                        help='List of files for LNEN dataset')
###############
# Evaluation 
###############
parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Parallelize the training on the data set')
parser.add_argument('--checkpoint_evaluation', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/train_tiles_harsh_dataaug_z128/checkpoint_30000.pth', type=Path,
                    metavar='DIR', help='path to checkpoint to evaluate')
parser.add_argument('--projector-dir', default='/gpfsscratch/rech/uli/ueu39kt/barlowtwins/projectors/train_tiles_harsh_dataaug_z128', type=Path,
                    metavar='DIR', help='path to where projectors will be saved')

##########
# Device
##########

parser.add_argument('--parallel', action='store_true', default=False,
                        help='Parallelize the training on the data set')



def train_loop(args, model, start_epoch, loader, optimizer, gpu, stats_file):
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        #sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                                lr_weights=optimizer.param_groups[0]['lr'],
                                lr_biases=optimizer.param_groups[1]['lr'],
                                loss=loss.item(),
                                time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
            if  step % args.save_freq == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}_{step}.pth')
                torch.save(model.backbone.state_dict(),
                       args.checkpoint_dir / f'wide_resnet50_{epoch}_{step}.pth')
        
        # save checkpoint
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}.pth')
        torch.save(model.backbone.state_dict(),
                args.checkpoint_dir / f'wide_resnet50_{epoch}.pth')
    
    # save final model
    torch.save(model.backbone.state_dict(),
                args.checkpoint_dir / 'wide_resnet50_final.pth')

def train_parallel_loop(args, model, start_epoch, loader, optimizer, gpu, stats_file):
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if idr_torch_rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            if idr_torch_rank == 0 and step % args.save_freq == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}_{step}.pth')
                torch.save(model.module.backbone.state_dict(),
                       args.checkpoint_dir / f'wide_resnet50_{epoch}_{step}.pth')
        if idr_torch_rank == 0 :
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / f'checkpoint_{epoch}.pth')
            torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / f'wide_resnet50_{epoch}.pth')
    if idr_torch_rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'wide_resnet50_final.pth')
    
def main():
    
    args = parser.parse_args()
    #################################
    #  Set computational environment 
    #################################
    
    # Parallel setting
    # ----------------
    if args.parallel:
        idr_torch_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        idr_world_size = int(os.environ['SLURM_NTASKS'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        torch.backends.cudnn.enabled = False

        # get node list from slurm
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)
    
        dist.init_process_group(backend='nccl', 
                                init_method='env://', 
                                world_size=idr_world_size, 
                                rank=idr_torch_rank)
    
        if idr_torch_rank == 0:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stats_file = open(args.checkpoint_dir / 'stats_eval.txt', 'a', buffering=1)
            print(' '.join(sys.argv))
            print(' '.join(sys.argv), file=stats_file)

        torch.cuda.set_device(local_rank)
        
        torch.backends.cudnn.benchmark = True
    
    else:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats_eval.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    gpu = torch.device("cuda")
    model = BarlowTwins(args).cuda(gpu)
    if args.parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    if args.parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        print("Lastest checkpoint loaded")
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    
    dataset = LNENDataset(args)
    print('Load LNEN data')
    
    if args.parallel:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        assert args.batch_size % idr_world_size == 0
        per_device_batch_size = args.batch_size // idr_world_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=per_device_batch_size, num_workers=0,
            pin_memory=True, sampler=sampler)
    else:
        loader =  torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    

    train_loop(args, model, start_epoch, loader, optimizer, gpu, stats_file)
   
        
def evaluate():
    args = parser.parse_args()
    idr_torch_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    idr_world_size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    torch.backends.cudnn.enabled = False

    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)
    
    dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_world_size, 
                            rank=idr_torch_rank)
    if idr_torch_rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    torch.cuda.set_device(local_rank)    
    torch.backends.cudnn.benchmark = True
    gpu = torch.device("cuda")

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    print("args.checkpoint_evaluation  ", args.checkpoint_evaluation)
    ckpt = torch.load(args.checkpoint_evaluation ,
                          map_location='cpu')
    print("**********************************************************")
    print("**********************************************************")
    print("automatically resume from checkpoint if it exists")
    print("**********************************************************")
    print("**********************************************************")
    
    optimizer.load_state_dict(ckpt['optimizer'])
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.load_state_dict(ckpt['model'])
    
    dataset = LNENDataset(args)
    print('Load LNEN data')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % idr_world_size == 0
    per_device_batch_size = args.batch_size // idr_world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=0,
        pin_memory=True, sampler=sampler)

    print('Len loader ', len(loader))
    scaler = torch.cuda.amp.GradScaler()
    model.eval()
    with torch.no_grad():
        for step, (y1, y2, path_to_imgs) in enumerate(loader):
            if step % 100 == 0:
                if idr_torch_rank == 0:
                    print('step ', step, 
                      '\n progression ' , (step ) /  len(loader))
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            z1, z2, loss = model.forward(y1, y2)
           
            if idr_torch_rank == 0:
                write_projectors(args, z1, path_to_imgs)
            
def write_projectors(args, z1, path_to_imgs):
    os.makedirs(os.path.join(args.projector_dir), exist_ok= True)   
    for i in range(len(path_to_imgs)):
        tne_id = path_to_imgs[i].split('/')[-3]
        os.makedirs(os.path.join(args.projector_dir, tne_id), exist_ok= True)
        img_name = path_to_imgs[i].split('/')[-1]
        z1_c = z1[i].squeeze().detach().cpu().numpy()
        np.save(os.path.join(args.projector_dir,tne_id,  img_name.split('.jpg')[0]), 
                z1_c)
        


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        main()
