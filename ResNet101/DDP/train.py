import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet101

# 设置分布式训练
def setup(rank, world_size):
    
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main(rank, world_size, args):
    
    setup(rank, world_size)

    # 数据集和数据加载器
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True)

    val_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 创建 resnet101 模型并使用 DDP 包装
    model = resnet101(num_classes=10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 优化器
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate)

    # 训练循环
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # 设置每个 epoch 的 sampler

        start_time = time.time()  # 记录开始时间
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and (i + 1) % 10 == 0:
                end_time = time.time()  # 记录结束时间
                throughput = 10 * train_loader.batch_size / (end_time - start_time)
                print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Throughput: {throughput:.2f} samples/s")
                start_time = end_time  # 更新开始时间为当前时间

        if rank == 0:
            accuracy = evaluate(ddp_model, val_loader, device=torch.device('cuda', rank))
            print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data-root', 
                        type=str, 
                        default='../DeepSpeed_ZeRO/tmp/cifar10-data',
                        help='Path to CIFAR-10 dataset')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=256, 
                        help='Batch size for training and validation')
    parser.add_argument('--learning-rate', 
                        type=float, 
                        default=0.001, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', 
                        type=int, 
                        default=10, 
                        help='Number of epochs to train')
    parser.add_argument('--world-size', 
                        type=int, 
                        default=1, 
                        help='Number of processes participating in the job')
    parser.add_argument('--rank', 
                        type=int, 
                        default=0, 
                        help='Rank of this process')
    args = parser.parse_args()

    # 如果是多GPU环境，需要通过torchrun或者类似的工具启动
    main(args.rank, args.world_size, args)