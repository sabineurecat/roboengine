import argparse
import os
import sys
import pdb

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoTokenizer
from utils.utils import dict_to_cuda
from functools import partial
import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from model.evf_sam2 import EvfSam2Model
from dataset_roboseg import RoboSegDataset, collate_fn, Resize


def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF finetune")
    parser.add_argument("--mode", type=str, default="robot", choices=["robot", "object", "task", "mix"])
    parser.add_argument("--eval_type", default="test", type=str, help="test or zero")
    parser.add_argument("--version_tokenizer", default="YxZhang/evf-sam2-multitask")
    parser.add_argument("--version", required=True)
    parser.add_argument("--image_size", default=224, type=int, help="image size")               
    parser.add_argument("--dataset_dir", default="../robo_seg_dataset/data", type=str)                      
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--renew_meta", action="store_true", default=False)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dist.init_process_group('nccl', init_method="env://")
    # rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.version_tokenizer, padding_side="right", use_fast=False)
    torch_dtype = torch.float32
    kwargs = {"torch_dtype":torch_dtype}
  
    model = EvfSam2Model.from_pretrained(args.version, **kwargs)
    model.eval()
    # model = DistributedDataParallel(model, device_ids=[rank])

    model.to(device)
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    test_dataset = RoboSegDataset(
        args.mode, args.eval_type, args.dataset_dir, args.image_size, transform = Resize(1024), 
        renew_meta=args.renew_meta,
    )
    # sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False, rank=rank)
    test_dataset = DataLoader(
        test_dataset, batch_size=1, 
        shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers,
        # sampler=sampler,
        collate_fn = partial(
            collate_fn, 
            tokenizer=tokenizer
        ),
    )

    giou, ciou = eval(test_dataset, model, args)
    print(f"model: {args.version}")
    print(f"ciou: {ciou}")
    print(f"giou: {giou}")
    # dist.destroy_process_group()
    print("Evaluation finished....")


def eval(test_loader, model, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    for imd, input_dict in tqdm.tqdm(enumerate(test_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        # print(f"After load dataset for val: Inference:{input_dict.get('inference')}")
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_evf"] = input_dict["images_evf"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()

        #########################
        os.makedirs("output", exist_ok=True)
        # pdb.set_trace()
        # Image.fromarray(((output_list[0] > 0).int().cpu().numpy()*255).astype(np.uint8)).save(f"output/{imd}_pred.png")
        # Image.fromarray((masks_list[0].cpu().numpy()*255).astype(np.uint8)).save(f"output/{imd}_mask.png")
        
        # pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(input_dict["images"].device)
        # pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(input_dict["images"].device)
        # img_TS = input_dict["images"] * pixel_std + pixel_mean
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[1].imshow((output_list[0] > 0).int().cpu().numpy())
        # axes[1].set_title("Prediction")
        # axes[1].axis('off')
        # axes[2].imshow(masks_list[0].cpu().numpy(), cmap='gray')
        # axes[2].set_title("GT Mask")
        # axes[2].axis('off')
        # axes[0].imshow(img_TS[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        # axes[0].set_title("Masked Image")
        # axes[0].axis('off')
        # plt.savefig(f"output/{imd}_vis.png", dpi=150)

        assert len(pred_masks) == 1
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            if output_i.shape != mask_i.shape:
                assert output_i.shape[0] == 1
                output_i = output_i[0]
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-10)
            acc_iou[union_i == 0] += 1.0  # no-object target

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0                   # iou calculate by the sum of intersection / union size from all images. 
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0   # average iou of each images.

    return giou, ciou

if __name__=="__main__":
    main(sys.argv[1:])    


# CUDA_VISIBLE_DEVICES=0 python eval.py --mode robot --eval_type test --version /cephfs/shared/yuanchengbo/roboengine/seg_ckpt/robot0115204622 --renew_meta