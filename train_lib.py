import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
import torch
from torch.optim import *
import matplotlib.pyplot as plt
from .utils import *
import time
import os
import shutil
from tqdm import tqdm
import logging
import cv2
import numpy as np

import urllib
import math
import itertools
from functools import partial
from torchmetrics import JaccardIndex


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output



def load_config_from_url(url: str) -> str:
    
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


TRAIN_MODEL_NAME = {
    # MODEL_NAME
    "unet": smp.Unet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    # "dinov2_vits14": torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'),
    # "dinov2_vitg14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc'),
    # "dinov2_vitg14_lc": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14'),
    # OPTIMIZER_NAME
    "SGD": SGD,
    "Adam": Adam,
    "RMSprop": RMSprop,
    "Adagrad": Adagrad,
    "Adadelta": Adadelta,
    # SCHEDULER_NAME
    "ConstantLR": lr_scheduler.ConstantLR,
    "StepLR": lr_scheduler.StepLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CyclicLR": lr_scheduler.CyclicLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    # CRITERION_NAME
    "JaccardLoss": JaccardLoss,
    "FocalLoss": FocalLoss,
    "DiceLoss": DiceLoss, 
    "SoftBCEWithLogitsLoss": SoftBCEWithLogitsLoss,
    "SoftCrossEntropyLoss": SoftCrossEntropyLoss
}


class TrainingModel:
    # opt needs to be the model itself in case of the SCHEDULER, or its parameters in the case of OPTIMIZER
    def __init__(self, cfg, opt=None, device=None):
        self.name = cfg.get("name")
        self.params = cfg.get("params")
        if opt:
            self.model = TRAIN_MODEL_NAME[self.name](opt, **self.params)
        else:
            if self.params:
                self.model = TRAIN_MODEL_NAME[self.name](**self.params)
            else:
                self.model = TRAIN_MODEL_NAME[self.name]
                # if self.name == 'dinov2_vitg14_lc':
                #     self.model.load_state_dict(torch.load("models/dinov2_vitg14_ade20k_m2f.pth"))
        if device:
            self.device = device


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = self.model.device
        if logger:
            self.logger = logger
            self.logger.wandb.watch(self.model.model, log_freq=20)

    def train_epoch(self, trainloader, valloader=None):
        # Training loop for one epoch
        
        self.model.model.to(device=self.device)
        self.model.model.train()
        temp_loss = 0
        # iou = JaccardIndex(task='binary').to(self.device)

        for data in trainloader:

            imgs = data['image'].to(device=self.device)
            masks = data['mask'].to(device=self.device)
            
            self.optimizer.model.zero_grad()
            pred_masks = self.model.model(imgs)
            loss = self.criterion.model(pred_masks, masks)
            loss.backward()
            self.optimizer.model.step()
            temp_loss += loss.item()
            # if valloader:
            #     iou(pred_masks, masks)
            #     val_loss = iou.compute()
            # if self.logger:
            # # save specific model for epoch if necessary
            # torch.save(self.model.model.state_dict(), 'models/training/model.pth')
        tot_loss = temp_loss/(len(trainloader))
        self.scheduler.model.step()
        
        # if valloader:
        #     return tot_loss, val_loss.item()
        return tot_loss

    def train(self, trainloader, epochs, valloader=None, testloader=None, NAME=None):
        # for idx in range(epochs):
        progress_bar = tqdm(total=epochs, desc="Training", unit="epoch", position=0, leave=True)
        for epoch in range(1, epochs + 1):

            loss = self.train_epoch(trainloader)
            # tqdm.write(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}", end='\r')
            
            if int(epoch) == int(epochs):
                # TIME = time.time()
                # torch.save(self.model.model.state_dict(), 'experiments/models/' + str(epoch) + '.pth')
                torch.save(self.model.model.state_dict(), 'experiments/models/' + str(NAME) + "_" + str(epoch) + '.pth')

            if valloader:
                valloss = self.evaluate(valloader)
            else:
                valloss = 0.0

            if testloader:
                testloss = self.evaluate(testloader)
            else:
                testloss = 0.0
            
            if hasattr(self, 'logger'):
                self.logger.wandb.log({
                    "Loss": loss,
                    "IoU Val": valloss,
                    "IoU Test": testloss,
                    "Learning rate": self.optimizer.model.param_groups[0]['lr']
                })
            progress_bar.set_postfix({"Loss": loss,
                                      "IoU Val": valloss,
                                      "IoU Test": testloss})#,
                                    #   "Test loss": testloss})
            progress_bar.update(1)
        progress_bar.close()

    def evaluate(self, valloader):
        self.model.model.to(device=self.device)
        self.model.model.eval()
        total_loss = 0.0
        iou = JaccardIndex(task='binary').to(self.device)
        
        with torch.no_grad():
            for data in valloader:
                imgs = data['image'].to(device=self.device)
                masks = data['mask'].to(device=self.device)
                
                preds = self.model.model(imgs)

                # loss = self.criterion.model(preds, masks)
                # if hasattr(self, 'logger'):
                #     wandb_str = "Jaccard Loss [" + wandb_option + "]"
                #     self.logger.wandb.log({
                #         wandb_str: loss
                #     })
                
                iou(preds, masks)
                loss = iou.compute()

                total_loss += loss.item()

        return (total_loss/len(valloader))
    
    def draw_and_save_preds(self, loader, cfg_path, exp_name, testing_set_name):
        # Create output directories
        base_output_dir = os.path.join('output', exp_name)
        timestamp_dir = os.path.join(base_output_dir, f"{testing_set_name}_{int(time.time())}")
        splash_dir = os.path.join(timestamp_dir, 'splash')
        masks_dir = os.path.join(timestamp_dir, 'masks')

        os.makedirs(splash_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # Copy config file
        shutil.copy(cfg_path, timestamp_dir)

        self.model.model.eval()
        with torch.no_grad():
            for data in loader:
                imgs = data['image'].to(device=self.device)
                preds = self.model.model(imgs)
                
                paths = data['path']
                
                for index, _ in enumerate(imgs):
                    path = paths[index]
                    img = imgs[index][0]
                    pred = preds[index][0]
                    pred = torch.sigmoid(pred)
                    pred = (pred >= 0.5).float()
                    
                    new_pred = pred.detach().cpu().numpy()
                    img = img.detach().cpu().numpy()
                    
                    filename = os.path.basename(path)
                    splash_file = os.path.join(splash_dir, filename)
                    
                    # Save splash image (mask over original)
                    draw_mask_over_image(path, img, new_pred, splash_file)

                    # Save binary mask
                    orig_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    orig_x, orig_y = orig_img.shape
                    stretched_pred = cv2.resize(new_pred, dsize=(orig_y, orig_x), interpolation=cv2.INTER_CUBIC)
                    
                    mask_file = os.path.join(masks_dir, filename)
                    cv2.imwrite(mask_file, (stretched_pred * 255).astype(np.uint8))

        logging.info(f"Saved predictions in {timestamp_dir}")
        return timestamp_dir

    def post_process_mask(self, mask):
        """Clean up the mask prediction."""
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def inference_save_direct(self, dataloader, masks_dir, splash_dir):
        """Run inference and save results directly to target directories."""
        self.model.model.eval()
        with torch.no_grad():
            for data in dataloader:
                imgs = data['image'].to(device=self.device)
                preds = self.model.model(imgs)
                paths = data['path']
                
                for index, _ in enumerate(imgs):
                    path = paths[index]
                    img = imgs[index][0]
                    pred = preds[index][0]
                    pred = torch.sigmoid(pred)
                    
                    # Get prediction as numpy array
                    pred_np = pred.detach().cpu().numpy()
                    img = img.detach().cpu().numpy()
                    
                    # Read original image to get dimensions
                    orig_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    orig_x, orig_y = orig_img.shape
                    
                    # Resize prediction using NEAREST interpolation to prevent smoothing
                    stretched_pred = cv2.resize(pred_np, dsize=(orig_y, orig_x), 
                                            interpolation=cv2.INTER_NEAREST)
                    
                    # Threshold and clean up the mask
                    binary_mask = (stretched_pred >= 0.5).astype(np.uint8)
                    processed_mask = self.post_process_mask(binary_mask)
                    
                    # Save files
                    filename = os.path.basename(path)
                    splash_file = os.path.join(splash_dir, filename)
                    mask_file = os.path.join(masks_dir, filename)
                    
                    # Save splash image
                    draw_mask_over_image(path, img, processed_mask, splash_file)
                    
                    # Save binary mask
                    cv2.imwrite(mask_file, (processed_mask * 255).astype(np.uint8))

    def freeze_smp_encoder(model):
        for child in model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False


    def unfreeze_smp_encoder(model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True


    
