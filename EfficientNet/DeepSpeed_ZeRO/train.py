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
import time
from torchvision.models import efficientnet_b0  # EfficientNet 替换 ResNet

dl_path = '../../ResNet50/DeepSpeed_ZeRO/tmp/cifar10-data/cifar-10-python'

def cifar_trainset(local_rank, dl_path=dl_path):
    transform = transforms.Compose(
    [
        transforms.Pad(4),  # 在图片四周填充4像素
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32),  # 随机裁剪出32x32的区域
        transforms.RandomRotation(15),  # 随机旋转，最大旋转角度为15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度、色调
        transforms.ToTensor(),  # 转换为Tensor格式，并将像素值缩放到[0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ]
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
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ]
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
    # 加载 EfficientNet-B0 模型
    net = efficientnet_b0(pretrained=False)  # 加载预训练权重
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 10)  # 修改为 CIFAR-10 的 10 类

    trainset = cifar_trainset(args.local_rank)
    valset = cifar_valset()

    engine, optimizer, dataloader, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    rank = dist.get_rank()
    criterion = torch.nn.CrossEntropyLoss()
    total_epochs = args.epochs
    # Get batch size from DeepSpeed config
    batch_size = engine.train_micro_batch_size_per_gpu()

    target_accuracy = 80.0
    start_time = time.time()  # 记录总训练开始时间
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()  # 记录每个epoch开始时间
        
        for batch in dataloader:
            inputs = batch[0].to(engine.device)
            labels = batch[1].to(engine.device)

            outputs = engine(inputs)
            loss = criterion(outputs, labels)
            engine.backward(loss)
            engine.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        it_per_sec = 1 / elapsed_time

        # 计算当前训练时间
        current_time = time.time() - start_time
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch time: {epoch_time:.2f}s, Total time: {current_time:.2f}s")
        # Validate and check accuracy
        val_loss, val_acc = validate(engine, valset, batch_size)
        print(f"Epoch {epoch + 1}/{total_epochs}: Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_acc:.4f}")
        
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Speed/iterations_per_sec", it_per_sec, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        
        # 检查是否达到目标准确率
        if val_acc >= target_accuracy:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\nReached target accuracy of {target_accuracy}%!")
            print(f"Total training time: {total_time:.2f} seconds")
            return  # 提前结束训练


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
