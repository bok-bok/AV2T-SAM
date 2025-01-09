import argparse
import random
import warnings
import wandb
import os
import logging

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel

import numpy as np
from model.avt_sam import AVTSAM
from utils.avs_dataset_m3 import MS3Dataset_SAM 
from utils.avs_dataset_s4 import S4Dataset_SAM
from utility import mask_iou, Eval_Fmeasure, AverageMeter

warnings.simplefilter("ignore", UserWarning)
criterionBCE = torch.nn.BCEWithLogitsLoss()

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


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)
    return iou.mean()

def compute_loss(pred_masks, masks_tensor):

    
    loss = criterionBCE(pred_masks, masks_tensor)
    iou_loss = _iou_loss(pred_masks, masks_tensor)
    return loss + iou_loss

def move_to_device(*tensors, device):
    return [tensor.to(device) for tensor in tensors]

def train(train_loader, model, optimizer: AdamW, args):
    model.train()
    device = args.device
    total_loss = 0
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    for batch in train_loader:
        sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor, original_size_list, _ = batch
        
        # Move tensors to the device in a single line
        sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor = move_to_device(
            sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor, device=device
        )
        
        B_i, T_i, C_i, H_i, W_i = sam_imgs_tensor.shape
        # B_m, T_m, C, H, W = masks_tensor.shape
        B, T, C, H, W = masks_tensor.shape
        masks_tensor = masks_tensor.view(B*T, C, H, W)

        
            

        optimizer.zero_grad()

        # Predict masks
        pred_masks = model(sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, original_size_list)
        # for S4 dataset
        # if T_i == 5 * T_m:
        #     # select first pred mask for each video
        #         indices = torch.tensor(list(range(0, len(pred_masks), 5)))
        #         indices = indices.cuda()

        #         pred_masks = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
        
        # Compute loss and backpropagate
        loss = compute_loss(pred_masks, masks_tensor)

        if torch.isnan(loss):
            print("NaN detected in loss, stopping training!")
            # Optionally save debug info here
            break  # break out of the inner loop

        loss.backward()
        total_loss += loss.item()

        # compute metrics
        pred_masks = pred_masks.view(B*T, H, W)
        masks_tensor = masks_tensor.view(B*T, H, W)

        
        
        miou = mask_iou(pred_masks, masks_tensor)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(pred_masks, masks_tensor)
        avg_meter_F.add({'F_score': F_score})
        
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Train Epoch: Loss: {avg_loss:.4f}")
    train_metrics = {
        'miou': avg_meter_miou.pop('miou').item(),
        'F_score': avg_meter_F.pop('F_score'),
        "train_loss": avg_loss
    }
    return train_metrics

@torch.no_grad()
def validate(val_loader, model, args):
    model.eval()
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    device = args.device
    total_loss = 0

    for batch in val_loader:
        sam_imgs_tensor, beit_imgs_tensor, clip_embeddings, clap_embeddings, masks_tensor, original_size_list, _ = batch
        
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

        # Compute loss for logging
        masks_tensor = masks_tensor.view(B*T, C, H, W)
        pred_masks = pred_masks.view(B*T, C, H, W)
        val_loss = compute_loss(pred_masks, masks_tensor)
        total_loss += val_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    # logging.info(f"Validation: Loss: {avg_loss:.4f}")
    # wandb.log({"val_loss": avg_loss})
        

    eval_metrics = {
        'miou': avg_meter_miou.pop('miou').item(),
        'F_score': avg_meter_F.pop('F_score'),
        "val_loss": avg_loss
    }
    logging.info(f"Validation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    return eval_metrics


@torch.no_grad()
def evaluate(test_loader, model, args):
    model.eval()
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    device = args.device
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



    eval_metrics = {
        'miou': avg_meter_miou.pop('miou').item(),
        'F_score': avg_meter_F.pop('F_score')
    }
    logging.info(f"Test Evaluation: mIoU: {eval_metrics['miou']:.4f}, F-score: {eval_metrics['F_score']:.4f}")
    return eval_metrics

def make_dataloader(args):
    assert args.dataset in ["ms3", "s4"], "Dataset not supported"
    if args.dataset == "ms3":
        train_loader = DataLoader(MS3Dataset_SAM(split='train', args = args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
        val_loader = DataLoader(MS3Dataset_SAM(split='val', args = args), batch_size=3, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        test_loader = DataLoader(MS3Dataset_SAM(split='test', args = args), batch_size=1, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    elif args.dataset == "s4":
        train_loader = DataLoader(S4Dataset_SAM(split='train', args = args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
        val_loader = DataLoader(S4Dataset_SAM(split='val', args = args), batch_size=3, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        test_loader = DataLoader(S4Dataset_SAM(split='test', args = args), batch_size=1, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def get_model(args) -> AVTSAM:
    # get model and move to device

    model = AVTSAM(args).to(device=args.device)

    gpu_count = torch.cuda.device_count()
    assert gpu_count in [1, 4], "Only 1 or 4 GPUs are supported"
    print(f"Using {gpu_count} GPUs")
    if gpu_count == 4:
        if args.evf_version == "evf_sam2":
            for i in range(48):
                if i < 5:
                    model.model.visual_model.image_encoder.trunk.blocks[i].to(f"cuda:3")
                elif i < 20:
                    model.model.visual_model.image_encoder.trunk.blocks[i].to(f"cuda:2")
                elif i < 39:
                    model.model.visual_model.image_encoder.trunk.blocks[i].to(f"cuda:1")
                else:
                    model.model.visual_model.image_encoder.trunk.blocks[i].to(f"cuda:0")
        else:
            for i in range(32):
                if i < 10:
                    model.model.visual_model.image_encoder.blocks[i].to(f"cuda:3")
                elif i < 20:
                    model.model.visual_model.image_encoder.blocks[i].to(f"cuda:2")
                elif i < 29:
                    model.model.visual_model.image_encoder.blocks[i].to(f"cuda:1")
                else:
                    model.model.visual_model.image_encoder.blocks[i].to(f"cuda:0")


    return model

def main(args):
    # Set up logging
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    setup_logger(os.path.join(args.save_path, "training_log.txt"))
    logging.info("Starting training process")
    max_val_v = 1e-8
    # name = f"{args.model_name}_{args.dataset}_{args.audio_aug}_{args.visual_aug}"

    # contain _adapter if using adapter
    name = f"{args.model_name}_{args.projector_type}{'_adapter' if args.use_adapter else ''}_{args.evf_version}_{args.dataset}_{args.audio_aug}_{args.visual_aug}"


    if args.wandb:
        wandb.init(project="avsbench", config=args,
                    name=name,
                )
    
    model: AVTSAM = get_model(args)
    
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.max_epochs, eta_min=args.min_lr)



    # Load data loaders
    train_loader, val_loader, test_loader = make_dataloader(args)

    for epoch in range(args.max_epochs):

        # Unfreeze decoder after 20 epochs
        # if ((args.max_epochs - epoch) + 1) == 20:
        #     model.set_decoder_trainable(True)

        logging.info(f"Epoch [{epoch + 1}/{args.max_epochs}]")
        # Train
        train_metrics = train(train_loader, model, optimizer, args)

        if args.wandb:
            wandb.log({"train_loss": train_metrics['train_loss'],
                    "train_miou": train_metrics['miou'],
                    "train_F_score": train_metrics['F_score']})
        scheduler.step()

        # Validate
        if epoch % 1 == 0:
            eval_result = validate(val_loader, model, args)
            if args.wandb:
                wandb.log({
                    "val_miou": eval_result['miou'],
                    "val_F_score": eval_result['F_score'],
                    "val_loss": eval_result['val_loss']
                })

            if eval_result['miou'] > max_val_v:
                max_val_v = eval_result['miou']
                model_save_path = os.path.join(args.save_path, name + "_best.pth")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best model to {model_save_path}")
    
    model_save_path = os.path.join(args.save_path, name + "_best.pth")
    model.load_state_dict(torch.load(model_save_path))
    test_result = evaluate(test_loader, model, args)
    if args.wandb:
        wandb.log({
            "test_miou": test_result['miou'],
            "test_F_score": test_result['F_score']
        })


if __name__ == "__main__":

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Set deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ms3", help="which subset of avsbench dataset")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--min_lr", type=float, default=1.0e-7)

    # Model 
    parser.add_argument("--save_path", type=str, default="/work/hdd/bcza/kb7180/weights")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--projector_type", type=str, default="default", help="Type of projector to use")
    parser.add_argument("--use_adapter", action="store_true", help="Use adapter layers")
    parser.add_argument("--adapter_type", type=str, default="clap", help="Type of adapter to use")
    parser.add_argument("--evf_version", type=str, default="evf_sam2", help="Which version of EVF to use")
    parser.add_argument("--av_fuse", action="store_true", help="Fuse audio and visual features")
    parser.add_argument("--cross_attn", action="store_true", help="Use cross attention")

    # training
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--normalize", action="store_true", help="normalize features")
    parser.add_argument("--augmentation", action="store_true", help="Use data augmentation")

    parser.add_argument("--visual_aug", type=float, default=0.0, help="Visual augmentation")
    parser.add_argument("--audio_aug", type=float, default=0.0, help="Audio augmentation")


    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
