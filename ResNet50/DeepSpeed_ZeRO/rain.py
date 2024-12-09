import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from deepspeed.utils import RepeatingLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet50
import time


dl_path = './tmp/cifar10-data'

def cifar_trainset(local_rank, dl_path=dl_path):
    transform = transforms.Compose(
        [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
    )
    # Ensure only one rank downloads.
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path, train=True, transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def cifar_valset(dl_path=dl_path):
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    valset = torchvision.datasets.CIFAR10(root=dl_path, train=False, transform=transform)
    return valset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    torch.manual_seed(args.seed)
    writer = SummaryWriter(log_dir='./runs')

    # Initialize model and data
    # 加载ResNet-50模型
    net = resnet50(num_classes=10)

    trainset = cifar_trainset(args.local_rank)
    valset = cifar_valset()

    engine, optimizer, dataloader, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    criterion = torch.nn.CrossEntropyLoss()
    total_epochs = args.epochs

    # Get batch size from DeepSpeed config
    batch_size = engine.train_micro_batch_size_per_gpu()

    for epoch in range(total_epochs):
        for step in range(len(trainset) // batch_size):
            start_time = time.time()
            batch = next(data_iter)
            inputs = batch[0].to(engine.device)
            labels = batch[1].to(engine.device)

            outputs = engine(inputs)
            loss = criterion(outputs, labels)
            engine.backward(loss)
            engine.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        it_per_sec = 1 / elapsed_time

        print(f"Epoch {epoch + 1}/{total_epochs}: {it_per_sec:.2f} it/s, Loss: {loss.item():.4f}")
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Speed/iterations_per_sec", it_per_sec, epoch)

        # Validate at the end of each epoch
        val_loss, val_acc = validate(engine, valset, batch_size)
        print(f"Epoch {epoch + 1}/{total_epochs}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)


def validate(engine, valset, batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(engine.device), batch[1].to(engine.device)
            outputs = engine.module(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


if __name__ == '__main__':
    args = get_args()

    # Initialize distributed mode
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    train_base(args)
