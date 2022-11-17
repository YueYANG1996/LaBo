"""
Linear probing following example in https://github.com/openai/CLIP
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
# import wandb

def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def get_img_feat(model, x, paper):
    model = model.visual
    x = model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([
        model.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
    ],
                  dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = model.ln_post(x[:, 0, :])
    if not paper:
        x = x @ model.proj
    return x

def get_features(dataloader, paper, clip_model='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device)
    convert_models_to_fp32(model)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # features = model.encode_image(images.to(device))
            if 'vit' not in clip_model.lower():
                features = model.visual(images.to(device))
            else:
                features = get_img_feat(model, images.to(device), paper)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(
        all_labels).cpu().numpy()


class LogisticRegression(pl.LightningModule):

    def __init__(self, n_cls, cfg) -> None:
        super().__init__()
        self.n_cls = n_cls
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, self.preprocess = clip.load(cfg.clip_model, device)
        convert_models_to_fp32(model)
        self.model = model
        self.paper = cfg.paper
        in_dim = 1024 if 'vit' not in cfg.clip_model.lower() else (768 if self.paper else 512)
        self.fc = nn.Linear(in_dim, n_cls)
        self.cfg = cfg
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        self.val_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        self.save_hyperparameters()

    def forward(self, x):
        return self.fc(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.lr,
                                     betas=(0.9, 0.98),
                                     eps=1e-6,
                                     weight_decay=0.2)
        return optimizer

    # def on_train_epoch_start(self, ):
    #     self.correct = 0
    #     self.cnt = 0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # x = self.preprocess(x)
        if 'vit' not in self.cfg.clip_model.lower():
            img_feat = self.model.visual(x)
        elif self.cfg.unfreeze_clip:
            img_feat = get_img_feat(self.model, x, self.paper)
        else:
            with torch.no_grad():
                img_feat = get_img_feat(self.model, x, self.paper)
        y_pred = self.forward(img_feat)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        # pred = y_pred.argmax(dim=-1)
        # self.correct += (pred == y).sum()
        # self.cnt += y.shape[0]
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    # def on_train_epoch_end(self,):
    #     # print(self.correct, self.cnt, self.correct / self.cnt)
    #     self.log('train_acc', self.correct/self.cnt)

    # def on_validation_epoch_start(self, ):
    #     self.val_correct = 0
    #     self.val_cnt = 0

    def validation_step(self, val_batch, batch_idx):
        if not self.cfg.DEBUG:
            # if self.global_step == 0 and not self.cfg.DEBUG:
            #     wandb.define_metric('val_acc', summary='max')
            if self.trainer.global_step == 0 and not self.cfg.DEBUG:
                import wandb
                wandb.define_metric('val_acc', summary='max')
        x, y = val_batch
        # x = self.preprocess(x)
        with torch.no_grad():
            if 'vit' not in self.cfg.clip_model.lower():
                img_feat = self.model.visual(x)
            else:
                img_feat = get_img_feat(self.model, x, self.paper)
        y_pred = self.forward(img_feat)
        loss = F.cross_entropy(y_pred, y)
        pred = y_pred.argmax(dim=-1)
        self.val_acc(y_pred, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        # self.val_correct += (pred == y).sum()
        # self.val_cnt += y.shape[0]
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            if 'vit' not in self.cfg.clip_model.lower():
                img_feat = self.model.visual(x)
            else:
                img_feat = get_img_feat(self.model, x, self.paper)
        y_pred = self.forward(img_feat)
        loss = F.cross_entropy(y_pred, y)
        self.test_acc(y_pred, y)
        self.log('test acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test loss', loss)
        return loss