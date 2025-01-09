import argparse
from PIL import Image
import random
import warnings
import logging
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.parallel

from model.avt_sam import AVTSAM
from utils.avs_dataset_m3 import MS3Dataset_SAM 
from utility import mask_iou, Eval_Fmeasure, AverageMeter

warnings.simplefilter("ignore", UserWarning)

# Setup logging
def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def move_to_device(*tensors, device):
    return [tensor.to(device) for tensor in tensors]

def save_mask(pred_masks,  video_name,args):
    save_base_path = args.save_path
    mask_save_path = os.path.join(save_base_path, video_name, f"pred_masks_{args.name}")


    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path, exist_ok=True)

    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    
    
    pred_masks = np.transpose(pred_masks, (1, 0, 2, 3))  # [5, 1, 224, 224]
    for idx in range(len(pred_masks)):
        one_mask = pred_masks[idx]

        # Squeeze the extra dimension so that one_mask has shape (224, 224)
        one_mask = np.squeeze(one_mask, axis=0)  # Shape becomes (224, 224)

        # Save the mask as a PNG image
        output_name = f"{idx}.png"
        im = Image.fromarray(one_mask).convert('L')  # 'L' mode for grayscale image
        im.save(os.path.join(mask_save_path, output_name), format='PNG')

def save_gt(gt_tensor, video_name, args):
    gt_tensor = gt_tensor.squeeze(0)
    gt = gt_tensor.cpu().data.numpy().astype(np.uint8)
    gt = gt * 255
    save_path = os.path.join(args.save_path, video_name, "gt")

    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    for i in range(gt.shape[0]):

        image_save_path = os.path.join(save_path, f"{i}.png")
        if os.path.exists(image_save_path):
            continue
        im = Image.fromarray(gt[i][0]).convert('L')
        im.save(image_save_path)

@torch.no_grad()
def evaluate(test_loader, model, args):
    model.eval()
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    device = args.device
    save_path = args.save_path
    for batch in test_loader:
        sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor, original_size_list, video_name = batch
        
        # Move tensors to the device in a single line
        sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor = move_to_device(
            sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor, device=device
        )

        B, T, C, H, W = masks_tensor.shape
        masks_tensor = masks_tensor.view(B*T, H, W)
        # Infer
        pred_masks = model(sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, original_size_list)

        pred_masks = pred_masks.view(B*T, H, W)

        # Compute metrics
        miou = mask_iou(pred_masks, masks_tensor)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(pred_masks, masks_tensor)
        avg_meter_F.add({'F_score': F_score})

        # Save masks and ground truth for this batch if saving is enabled
        if args.save:
            video_name = video_name[0]
            save_gt(masks_tensor.view(B, T, 1, H, W), video_name, args)
            save_mask(pred_masks.view(B, T, 1, H, W), video_name, args)

    eval_metrics = {
        'miou': avg_meter_miou.pop('miou').item(),
        'F_score': avg_meter_F.pop('F_score')
    }
    logging.info(f"Test Evaluation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    return eval_metrics

def main(args):
    # Set up logging
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    setup_logger(os.path.join(args.save_path, f"test_log_{args.name}.txt"))
    logging.info("Starting evaluation process")
    
    model = AVTSAM(args).to(device=args.device)
    if args.weight_path:
        print(f"Loading weights from {args.weight_path}")
        model.load_state_dict(torch.load(args.weight_path), strict=False)
        
    
    

    # Parallelize model if needed
    test_loader = DataLoader(MS3Dataset_SAM(split='test', args = args), batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Evaluate
    eval_result = evaluate(test_loader, model, args)
    logging.info({
        "miou": eval_result['miou'],
        "F_score": eval_result['F_score']
    })

if __name__ == "__main__":

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ms3", help="which subset of avsbench dataset")

    # Model parameters
    parser.add_argument("--save_path", type=str, default="tmp")
    # evaluation
    parser.add_argument("--parallel", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--save", action="store_true", help="Save the segmentation masks and ground truth")
    parser.add_argument("--weight_path", type=str, default="weights/model_best.pth", help="Path to the model weights")
    parser.add_argument("--name", type=str, default="sample", help="Name of the experiment")


    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--projector_type", type=str, default="default", help="Type of projector to use")
    parser.add_argument("--use_adapter", action="store_true", help="Use adapter layers")
    parser.add_argument("--adapter_type", type=str, default="clap", help="Type of adapter to use")
    parser.add_argument("--evf_version", type=str, default="evf_sam2", help="Which version of EVF to use")
    parser.add_argument("--av_fuse", action="store_true", help="Fuse audio and visual features")


    parser.add_argument("--normalize", action="store_true", help="normalize features")
    parser.add_argument("--augmentation", action="store_true", help="Use data augmentation")


    parser.add_argument("--visual_aug", type=float, default=0.0, help="Visual augmentation")
    parser.add_argument("--audio_aug", type=float, default=0.0, help="Audio augmentation")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
# weights/model_best.pth