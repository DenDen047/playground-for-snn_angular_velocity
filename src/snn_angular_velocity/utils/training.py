import numpy as np
import os
import torch
from tqdm import tqdm

from data_loader.training import TrainDatabase
from data_loader.validating import ValDatabase
from model.metric import medianRelativeError, rmse

from .gpu import moveToGPUDevice
from .tbase import TBase

from model.loss import compute_loss


class Trainer(TBase):
    def __init__(self, data_dir, write, log_config, general_config):
        super().__init__(data_dir, log_config, general_config)
        self.write_output = write
        self.output_dir = self.log_config.getOutDir()

        train_database = TrainDatabase(self.data_dir)
        val_database = ValDatabase(self.data_dir)
        self.batchsize = general_config['batchsize']
        self.train_loader = torch.utils.data.DataLoader(
                train_database,
                batch_size=self.batchsize,
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(
                val_database,
                batch_size=self.batchsize,
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)

        self.data_collector = DataCollector(general_config['simulation']['tStartLoss'])

    def train(self):
        self._loadNetFromCheckpoint()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, amsgrad=False)

        i = 0
        iterations = 240000
        while i < iterations:
            print('iteration: {}'.format(i))

            # train
            self.net = self.net.train()
            for data in tqdm(self.train_loader, desc='training'):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']
                ang_vel_gt = data['angular_velocity']

                optimizer.zero_grad()

                ang_vel_pred = self.net(spike_tensor)

                loss = compute_loss(ang_vel_pred, ang_vel_gt)
                loss.backward()

                optimizer.step()

                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])

            i += self.batchsize

            # # val
            # self.net = self.net.eval()
            # for data in tqdm(self.val_loader, desc='val in training'):
            #     data = moveToGPUDevice(data, self.device, self.dtype)

            #     spike_tensor = data['spike_tensor']
            #     ang_vel_gt = data['angular_velocity']

            #     ang_vel_pred = self.net(spike_tensor)
            #     self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])

            # if self.write_output:
            #     self.data_collector.writeToDisk(self.output_dir)
            self.data_collector.printErrors()


class DataCollector:
    def __init__(self, loss_start_idx: int):
        assert loss_start_idx >= 0
        self.loss_start_idx = loss_start_idx

        self.file_indices = list()
        self.data_gt = list()
        self.data_pred = list()

    def append(self, pred: torch.Tensor, gt: torch.Tensor, file_indices: torch.Tensor):
        # pred/gt: (batchsize, 3, time) tensor
        # file_indices: (batchsize) tensor

        self.data_pred.append(pred.detach().cpu().numpy())
        self.data_gt.append(gt.detach().cpu().numpy())
        self.file_indices.append(file_indices.to(torch.long).cpu().numpy())

    def writeToDisk(self, out_dir: str):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)
        file_indices = np.concatenate(self.file_indices)
        np.save(os.path.join(out_dir, 'predictions.npy'), pred)
        np.save(os.path.join(out_dir, 'groundtruth.npy'), gt)
        np.save(os.path.join(out_dir, 'indices.npy'), file_indices)

    def printErrors(self):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)

        pred = pred[..., self.loss_start_idx:]
        gt = gt[..., self.loss_start_idx:]

        print('RMSE: {} deg/s'.format(rmse(pred, gt, deg=True)))
        print('median of relative error: {}'.format(medianRelativeError(pred, gt)))
