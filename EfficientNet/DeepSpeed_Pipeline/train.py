import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from torchvision.models import efficientnet_b0  # 使用EfficientNet-B0
import time

dl_path = '../../ResNet50/DeepSpeed_ZeRO/tmp/cifar10-data/cifar-10-python'

def cifar_trainset(local_rank, dl_path=dl_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整到 EfficientNet 的输入分辨率
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),  # 确保随机裁剪后的分辨率正确
            transforms.ToTensor(),
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
            transforms.Resize((224, 224)),  # 调整到 EfficientNet 的输入分辨率
            transforms.ToTensor(),
        ]
    )
    valset = torchvision.datasets.CIFAR10(root=dl_path, train=False, transform=transform)
    return valset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', 
                        type=int, 
                        default=1138, 
                        help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    torch.manual_seed(args.seed)

    # 使用 EfficientNet-B0
    net = efficientnet_b0(pretrained=False)  # 加载预训练权重
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 10)  # 修改为 CIFAR-10 的 10 类


    trainset = cifar_trainset(args.local_rank)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')

# 修改 join_layers 函数以适应 EfficientNet
def join_layers(vision_model):
    layers = [
        vision_model.features,  # EfficientNet特有的特征提取部分
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        vision_model.classifier,
    ]
    return layers

def validate(engine, valset, batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=True)
    data_iter = iter(dataloader)
    
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:  # 遍历数据批次
            labels = batch[1].to(engine.device)  # 获取 inputs 和 labels
            loss, outputs = engine.eval_batch(data_iter=data_iter, return_logits=True)  # 让 eval_batch 自行取数据
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    writer = SummaryWriter(log_dir='./runs')
    #
    # Build the model
    #

    # 使用 EfficientNet-B0
    net = efficientnet_b0(pretrained=True)  # 加载预训练权重
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 10)  # 修改为 CIFAR-10 的 10 类

    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)
    valset = cifar_valset()

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    for epoch in range(args.epochs):
        for step in range(len(trainset) // engine.train_micro_batch_size_per_gpu()):    
            start_time = time.time()
            loss = engine.train_batch()
            end_time = time.time()
            elapsed_time = end_time - start_time
            it_per_sec = 1 / elapsed_time

            writer.add_scalar("Learning Rate", engine.optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Speed/iterations_per_sec", it_per_sec, step)
            
            if step % 10 == 0:
                print(f"Iteration {step} (Epoch {epoch+1}): {it_per_sec} it/s")

        
        val_loss, val_acc = validate(engine, valset, engine.train_micro_batch_size_per_gpu())
        print(f"Epoch {epoch + 1}/{args.epochs}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)

        if epoch % 5 == 0:  # 每 5 个 epoch 保存一次检查点
            engine.save_checkpoint("./tmp", tag=f"epoch_{epoch}")


if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)