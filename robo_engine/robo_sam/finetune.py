import argparse
import os
import sys
import pdb

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils.utils import dict_to_cuda
from functools import partial
import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


from model.evf_sam2 import EvfSam2Model
from dataset_roboseg import RoboSegDataset, collate_fn, Resize


def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF finetune")
    parser.add_argument("--mode", type=str, default="task", choices=["robot", "object", "task", "mix"])
    parser.add_argument("--base_version", type=str, default="YxZhang/evf-sam2-multitask")
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--dataset_dir", default="../robo_seg_dataset/data", type=str)                      
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--renew_meta", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=float, default=100)
    parser.add_argument("--patience", type=float, default=10)
    parser.add_argument("--save_dir", type=str, default="/cephfs/shared/yuanchengbo/roboengine/seg_ckpt")
    parser.add_argument("--commit", type=str, default="")
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dist.init_process_group("nccl", init_method="env://")
    # rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained("YxZhang/evf-sam2", padding_side="right", use_fast=False)
    torch_dtype = torch.float32
    kwargs = {"torch_dtype":torch_dtype}
  
    model = EvfSam2Model.from_pretrained(args.base_version)
    model.enable_train()
    # model = DistributedDataParallel(model, device_ids=[rank])

    model.to(device)
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    train_dataset = RoboSegDataset(
        args.mode, "train", args.dataset_dir, args.image_size, transform = Resize(1024), 
        renew_meta=args.renew_meta,
        apply_base_augmentation=True,
    )

    train_dataset.__getitem__(0)

    # sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, 
        shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers,
        # sampler=sampler,
        collate_fn = partial(
            collate_fn, 
            tokenizer=tokenizer
        ),
    )

    val_dataset = RoboSegDataset(
        args.mode, "val", args.dataset_dir, args.image_size, transform = Resize(1024),
        renew_meta=args.renew_meta,
    )
    # sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False, rank=rank)
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, 
        shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers,
        # sampler=sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            # local_rank=rank,
        ),
    )

    now = datetime.now()    
    uid = now.strftime("%m%d%H%M%S")
    args.commit = args.mode + uid 

    train(train_loader,val_loader, model, args)
    # dist.destroy_process_group()
    print("Training finished....")


def train(train_loader,val_loader, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.data.mean()}")

    num_epochs = args.num_epochs
    patience = args.patience
    min_val_loss = float("inf")
    early_stopping_hook = 0
    tracking_information = []
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for input_dict in tqdm.tqdm(train_loader):
            input_dict['inference']=False
            # print(f"After load dataset for train: Inference:{input_dict.get('inference')}")
            input_dict = dict_to_cuda(input_dict)
            output_dict = model(**input_dict)
            loss = output_dict.get("loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()    

        val_loss=0
        model.eval()
        for input_dict in tqdm.tqdm(val_loader):
            input_dict['inference']=False
            # print(f"After load dataset for val: Inference:{input_dict.get('inference')}")
            input_dict = dict_to_cuda(input_dict)
            output_dict = model(**input_dict)
            loss = output_dict.get("loss")
            val_loss += loss.item() 

        tracking_information.append((train_loss/len(train_loader), val_loss/len(val_loader), optimizer.param_groups[0]["lr"]))
        print(f"Epoch : {epoch} Training loss: {train_loss/len(train_loader)} Eval loss: {val_loss/len(val_loader)} LR: {optimizer.param_groups[0]['lr']} Best: {min_val_loss/len(val_loader)}")
        scheduler.step()

        if val_loss < min_val_loss:
            model.save_pretrained(f"{args.save_dir}/{args.commit}",from_pt=True)
            print(f"Saved model to {args.save_dir}/{args.commit}")
            min_val_loss = val_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook+=1
            if (early_stopping_hook>patience):
                break 

if __name__=="__main__":
    main(sys.argv[1:])    


# CUDA_VISIBLE_DEVICES=0 python finetune.py --mode robot --lr 1e-5 --patience 15 --renew_meta

# https://spacy.io/models/en#en_core_web_sm
# https://github.com/facebookresearch/sam2/issues/48#issuecomment-2259542252 