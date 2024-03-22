import os

os.environ['CUDA_VISIBLE_DEVICES']='2, 3, 4, 5'

import warnings

import torch
import torch.nn as nn
from models import DiscreteDiffusion
from process import *
from torch import optim
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


class DiscreteDiffusionTrainer(Trainer):
    def run_epoch(self, epoch, loader, train:bool):
        loss_per_epoch = 0.0
        if train:
            self.model.train()
            if epoch < self.warmup_epochs:
                lr_scale = (float(epoch + 1) / float(self.warmup_epochs)) ** 4
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = lr_scale * self.max_lr[idx]

        else:
            self.model.eval()

        for i, data in enumerate(loader):
            seq, seq_mask, sec, sec_mask = move_to_device(data, self.device)

            with torch.set_grad_enabled(train):
                # loss = self.model(seq, sec, sec_mask)
                loss = self.model(seq, ss=sec, padding_mask=sec_mask)


                if train:
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    # loss.backward()
                    self.optimizer.step()

            loss_per_epoch += loss.item()
        loss = loss_per_epoch / len(loader)

        if train:
            self.train_loss.append(loss)
        else:
            self.valid_loss.append(loss)
            self.scheduler.step(loss)
        return loss


def run(config):
    seed_torch(config.seed)
    # for generation training
    dataset = SequenceDataset(
        csv_path = config.dataset.csv_path,
        min_len  = config.dataset.min_len,
        max_len  = config.dataset.max_len,
    )
    train_data, valid_data, _ = split_dataset(dataset, train_percent=0.8, seed=config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size = config.batch_size,
        shuffle    = True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size = config.batch_size,
        shuffle    = False
    )

    # for generation training
    diffusion = DiscreteDiffusion(
        num_steps        = config.model.num_steps,
        num_classes      = config.model.num_classes,
        schedule         = config.model.schedule,
        transition_type  = config.model.transition_type,
        d_model          = config.model.emb_dim,
        num_heads        = config.model.num_heads,
        num_layers       = config.model.num_layers,
        max_seq_length   = config.model.seq_len,
        loss_type        = config.model.loss_type
    )

    optimizer = optim.Adam(diffusion.parameters(), lr=config.scheduler.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = config.scheduler.mode,
        factor   = config.scheduler.factor,
        cooldown = config.scheduler.cooldown,
        min_lr   = config.scheduler.min_lr
    )
    # for generation training
    trainer = DiscreteDiffusionTrainer(
        model         = diffusion,
        train_loader  = train_loader,
        valid_loader  = valid_loader,
        optimizer     = optimizer,
        ckpt_path     = config.cache,
        resume        = config.resume,
        scheduler     = scheduler,
        warmup_epochs = 20,
        seed          = config.seed
    )

    trainer.logger.info(config)
    trainer.train(config.epoch)
    print('Finished training ')


if __name__ == "__main__":
    config = get_config('./config')
    run(config)
