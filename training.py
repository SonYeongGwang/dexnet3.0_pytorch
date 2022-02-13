import argparse
import os
import sys
import datetime
import copy

import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device_count

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import mnist

from dsets import SuctionDataset, TrainingSuctionDataset, ValidationSuctionDataset
from model import SuctionModel

writer = SummaryWriter("runs")

class SuctionTraining:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='Number of worker processes for background data loading', default=20, type=int, )
        parser.add_argument('--batch-size', help='Batch size to use for training', default=64, type=int, )
        parser.add_argument('--epochs', help='Number of epochs to train for', default=25, type=int,)

        self.cli_args = parser.parse_args(sys_argv)
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    ## initiate the model
    ## use multiple GPUs if possible
    def initModel(self):
        model = SuctionModel()
        if self.use_cuda:
            print("INFO: Using CUDA; {} device(s).".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                print("INFO: Trainin with multiple devices.")
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initDataContainer(self, number_of_files):
        return SuctionDataset(number_of_files=number_of_files)

    def initTrainDl(self, raw_dataset):
        train_ds = TrainingSuctionDataset(raw_dataset, val_stride=3) # 25% of validation

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda, shuffle=True)

        return train_dl

    def initValDl(self, raw_dataset):
        val_ds = ValidationSuctionDataset(raw_dataset, val_stride=3) # 25% of validation

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

        return val_dl

    def training_loop(self):
        self.train_loss = []
        self.validation_loss = []
        self.train_acc = []
        self.validation_acc = []

        ## DexNet3.0 datasets consist of 2,760 npz files
        self.data_container = self.initDataContainer(number_of_files=2760)
        self.train_dl = self.initTrainDl(self.data_container)
        self.val_dl = self.initValDl(self.data_container)
        optimizer = self.initOptimizer()
        loss_fn = nn.CrossEntropyLoss()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            loss_train = 0.0
            acc_train = 0.0
            total_train = 0
            total_val = 0

            print("INFO: Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(self.train_dl),
                len(self.val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1)
            ))
            for imgs, hand_pose, labels in self.train_dl:
                imgs = imgs.to(device=self.device)
                hand_pose = hand_pose.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self.model(imgs, hand_pose)

                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train += len(labels)
                loss_train += loss.item()
                acc_train += (int((outputs.argmax(dim=1) == labels).sum()))

            with torch.no_grad():
                loss_val = 0.0
                acc_val = 0.0
                for imgs, hand_pose, labels in self.val_dl:
                    imgs = imgs.to(device=self.device)
                    hand_pose = hand_pose.to(device=self.device)
                    labels = labels.to(device=self.device)
                    outputs = self.model(imgs, hand_pose)
                    loss = loss_fn(outputs, labels)

                    total_val += len(labels)
                    loss_val += loss.item()
                    acc_val += (int((outputs.argmax(dim=1) == labels).sum()))

            writer.add_scalars("Loss", {"training": loss_train/len(self.train_dl), "validation": loss_val/len(self.val_dl)}, epoch_ndx)
            writer.add_scalars("Accuracy", {"training": acc_train/total_train*100, "validation": acc_val/total_val*100}, epoch_ndx)

            self.train_loss.append(loss_train/len(self.train_dl))
            self.train_acc.append(acc_train/total_train*100)
            self.validation_loss.append(loss_val/len(self.val_dl))
            self.validation_acc.append(acc_val/total_val*100)
            print('{} Epoch {}, Training loss {}, Validation loss {}'.format(datetime.datetime.now(), epoch_ndx,
                loss_train / len(self.train_dl), loss_val/len(self.val_dl)))
            print('{} Training accuracy {}, Validation accuracy {}'.format(datetime.datetime.now(),
                acc_train/total_train*100, acc_val/total_val*100))
            print("-"*100)

        plt.figure("Loss")
        plt.plot(np.arange(self.cli_args.epochs), self.train_loss, label="Train_Loss")
        plt.plot(np.arange(self.cli_args.epochs), self.validation_loss, label="Validation_Loss")
        plt.xlabel("n_epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.figure("Validation")
        plt.plot(np.arange(self.cli_args.epochs), self.train_acc, label="Train_Accuary")
        plt.plot(np.arange(self.cli_args.epochs), self.validation_acc, label="Validation_Accuary")
        plt.xlabel("n_epoch")
        plt.ylabel("Accuary")
        plt.legend()

        plt.show()
        writer.close()

    def validate(self):
        for name, loader in [("train", self.train_dl), ("val", self.val_dl)]:
            correct = 0
            total = 0

            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device=self.device)
                    labels = labels.to(device=self.device)
                    outputs = self.model(imgs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())

            print("Accuracy {}: {:.2f}".format(name, correct/total))


if __name__ == "__main__":
    ST = SuctionTraining()
    ST.training_loop()