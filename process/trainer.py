import logging
import os
import shutil
import time
from abc import ABC, abstractclassmethod

import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import *

# class Trainer:
#     def __init__(self, model, train_loader, valid_loader, device, optimizer=None, loss_fn=None,
#                  ckpt_path=None, resume=False, scheduler=None, warmup_epochs=0, seed=0):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.valid_loader = valid_loader
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.loss_fn = loss_fn
#         self.device = device
#         self.ckpt_path = ckpt_path
#         self.resume = resume
#         self.start_epoch = 0
#         self.train_loss = []
#         self.valid_loss = []
#         self.warmup_epochs = warmup_epochs
#         self.max_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
#         self.seed = seed

#         self.start_time = time.strftime("%m-%d-%H", time.localtime())
#         self._init_file()
#         logging.basicConfig(
#             filename=f'./cache/{self.start_time}/train.log', level=logging.INFO,
#             format='{} {} {}\n{}'.format('%(asctime)s', '%(levelname)s', ':', '%(message)s'),
#             datefmt='%Y-%m-%d %H:%M:%S'
#         )
#         self.logger = logging.getLogger(__name__)

#         self._init_tensorboard()

#         if self.resume and self.ckpt_path is not None:
#             checkpoint_path = f'./cache/{self.ckpt_path}/latest.pt'
#             if os.path.isfile(checkpoint_path):
#                 self._load_checkpoint(checkpoint_path)
#                 print(f"Resuming from checkpoint {checkpoint_path}")
#             else:
#                 print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")

#     def _init_file(self):
#         os.makedirs(f'./cache/{self.start_time}', exist_ok=True)
#         os.makedirs(f'./cache/{self.start_time}/logs', exist_ok=True)
#         shutil.copytree('./models', f'./cache/{self.start_time}/models', dirs_exist_ok=True)

#     def _init_fn(self, worker_id):
#         np.random.seed(int(self.seed) + worker_id)

#     def _init_tensorboard(self):
#         self.writer = SummaryWriter(log_dir=f'./cache/{self.start_time}/logs')

#     def _load_checkpoint(self, path):
#         checkpoint = torch.load(path)
#         self.model.load_state_dict(checkpoint['model'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         self.scheduler.load_state_dict(checkpoint['scheduler'])
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.train_loss = checkpoint['train_loss']
#         self.valid_loss = checkpoint['valid_loss']

#     def _save_checkpoint(self, epoch, tag):
#         checkpoint = {
#             'epoch': epoch,
#             'model': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'scheduler':self.scheduler.state_dict(),
#             'train_loss': self.train_loss,
#             'valid_loss': self.valid_loss,
#         }
#         torch.save(checkpoint, f'./cache/{self.start_time}/{tag}.pt')

#     @abstractclassmethod
#     def run_epoch(self, epoch, loader, train:bool):
#         pass

#     def train(self, num_epochs):
#         for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
#             train_loss = self.run_epoch(epoch, self.train_loader, train=True)
#             valid_loss = self.run_epoch(epoch, self.valid_loader, train=False)

#             train_item = retrieve_name([train_loss])
#             valid_item = retrieve_name([valid_loss])

#             print(f'epoch:{epoch}||train_loss:{train_loss:.4f}||valid_loss:{valid_loss:.4f}', flush=True)

#             self.scheduler.step(valid_loss)

#             self.writer = tensorboard_writer(self.writer, epoch, [train_item, valid_item])
#             self.logger.info(f"Learning Rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
#             self.logger.info(f'Train Loss: {train_loss:4f}, Valid Loss: {valid_loss:4f}')

#             self._save_checkpoint(epoch, 'latest')
#             if epoch == self.start_epoch or valid_loss < min(self.valid_loss):
#                 self._save_checkpoint(epoch, 'best')
#         self.writer.close()


class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer=None, loss_fn=None,
                 scheduler=None, ckpt_path=None, resume=False, warmup_epochs=0, seed=0):
        # Initialize accelerator
        self.accelerator = Accelerator(cpu=False, mixed_precision='fp16', device_placement=True)
        self.accelerator.print(f'device {str(self.accelerator.device)} is used!')

        # Set the device provided by accelerator
        self.device = self.accelerator.device

        # Prepare everything
        self.model = self.accelerator.prepare(model)
        self.optimizer = self.accelerator.prepare(optimizer)
        self.loss_fn = self.accelerator.prepare(loss_fn)
        self.scheduler = self.accelerator.prepare(scheduler)
        self.train_loader = self.accelerator.prepare(train_loader)
        self.valid_loader = self.accelerator.prepare(valid_loader)

        self.start_epoch = 0
        self.train_loss = []
        self.valid_loss = []
        self.warmup_epochs = warmup_epochs
        self.max_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.seed = seed

        self.start_time = time.strftime("%m-%d-%H", time.localtime())
        self._init_file()
        self._init_logger()
        self._init_tensorboard()

        if resume and ckpt_path is not None:
            checkpoint_path = f'./cache/{ckpt_path}/latest.pt'
            if os.path.isfile(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
                print(f"Resuming from checkpoint {checkpoint_path}")
            else:
                print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")

    def _init_logger(self):
        logging.basicConfig(
            filename=f'./cache/{self.start_time}/train.log', level=logging.INFO,
            format='{} {} {}\n{}'.format('%(asctime)s', '%(levelname)s', ':', '%(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def _init_file(self):
        os.makedirs(f'./cache/{self.start_time}', exist_ok=True)
        os.makedirs(f'./cache/{self.start_time}/logs', exist_ok=True)
        shutil.copytree('./models', f'./cache/{self.start_time}/models', dirs_exist_ok=True)
        shutil.copytree('./config', f'./cache/{self.start_time}/config', dirs_exist_ok=True)

    def _init_fn(self, worker_id):
        np.random.seed(int(self.seed) + worker_id)

    def _init_tensorboard(self):
        if self.accelerator.is_local_main_process:
            self.writer = SummaryWriter(log_dir=f'./cache/{self.start_time}/logs')

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path)

        # Loading the model state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        # self.train_loss = checkpoint['train_loss']
        # self.valid_loss = checkpoint['valid_loss']

    def _save_checkpoint(self, epoch, tag):
        # Check if this process should save the checkpoint (for distributed training)
        if self.accelerator.is_local_main_process:
            checkpoint = {
                'epoch': epoch,
                'model': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'train_loss': self.train_loss,
                'valid_loss': self.valid_loss,
            }
            torch.save(checkpoint, f'./cache/{self.start_time}/{tag}.pt')

    def determine_max_batch_size(self, initial_batch_size):
        def model_fn(batch_size):
            # Adjust this based on how your dataset and DataLoader are set up
            temp_loader = DataLoader(self.train_loader.dataset, batch_size=batch_size, shuffle=True)
            input_data = next(iter(temp_loader))
            input_data = self.accelerator.prepare(input_data)  # Prepare the data using accelerator
            output = self.model(*input_data)
            return output

        max_batch_size = find_executable_batch_size(model_fn, starting_batch_size=initial_batch_size)
        return max_batch_size

    @abstractclassmethod
    def run_epoch(self, epoch, loader, train:bool):
        pass

    def train(self, num_epochs):
        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            train_loss = self.run_epoch(epoch, self.train_loader, train=True)
            valid_loss = self.run_epoch(epoch, self.valid_loader, train=False)
            lr         = self.optimizer.state_dict()['param_groups'][0]['lr']

            train_item = retrieve_name([train_loss])
            valid_item = retrieve_name([valid_loss])
            lr_item    = retrieve_name([lr])

            if self.accelerator.is_local_main_process:
                self.writer = tensorboard_writer(self.writer, epoch, [train_item, valid_item])
                self.writer = tensorboard_writer(self.writer, epoch, lr_item)
                self.logger.info(f"Learning Rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
                self.logger.info(f'EPOCH: {epoch}, Train Loss: {train_loss:4f}, Valid Loss: {valid_loss:4f}')
                if (epoch+1) % 10 == 0 or epoch+1 == num_epochs:
                    self._save_checkpoint(epoch, 'latest')
        if self.accelerator.is_local_main_process:
            self.writer.close()
