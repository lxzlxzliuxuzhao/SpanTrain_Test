#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from torchvision.models import resnet50
import time

dl_path = '../DeepSpeed_ZeRO/tmp/cifar10-data'

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

class PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 计算需要填充的数量
        self.pad_size = (batch_size - len(dataset) % batch_size) % batch_size
        self.length = len(dataset) + self.pad_size
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            # 循环使用数据集开头的数据进行填充
            return self.dataset[idx % len(self.dataset)]


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

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = resnet50(num_classes=10)

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

# 修改 join_layers 函数以适应 ResNet50
def join_layers(vision_model):
    layers = [
        vision_model.conv1,
        vision_model.bn1,
        vision_model.relu,
        vision_model.maxpool,
        *vision_model.layer1,
        *vision_model.layer2,
        *vision_model.layer3,
        *vision_model.layer4,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        vision_model.fc,
    ]
    return layers

def validate(engine, valset, batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=True)
    data_iter = iter(dataloader)
    
    # 只在最后阶段初始化计数器
    if engine.is_last_stage():
        correct = 0
        total = 0
        total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(engine.device), batch[1].to(engine.device)
            loss, outputs = engine.eval_batch(data_iter=data_iter, return_logits=True)
            
            # 只在最后阶段计算损失和准确率
            if engine.is_last_stage():
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    # 只在最后阶段计算最终结果
    if engine.is_last_stage():
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    else:
        # 非最后阶段返回None
        return None, None
    


def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    writer = SummaryWriter(log_dir='./runs')
    #
    # Build the model
    #

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = resnet50(num_classes=10)
    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)
    trainset = PaddedDataset(trainset, 256)
    valset = cifar_valset()
    
    # 记录训练开始时间
    start_time = time.time()
    target_accuracy = 80.0
    reached_target = False
    
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)
    
    total_step = 0
    for epoch in range(args.epochs):
        for step in range(len(trainset) // engine.train_micro_batch_size_per_gpu()):    
            start_time_batch = time.time()
            loss = engine.train_batch()
            end_time_batch = time.time()
            elapsed_time_batch = end_time_batch - start_time_batch
            it_per_sec = 1 / elapsed_time_batch

            writer.add_scalar("Learning Rate", engine.optimizer.param_groups[0]["lr"], total_step)
            writer.add_scalar("Loss/train", loss.item(), total_step)
            writer.add_scalar("Speed/iterations_per_sec", it_per_sec, total_step)
            
            total_step += 1
            
            if step % 10 == 0:
                print(f"Iteration {step} (Epoch {epoch+1}): {it_per_sec} it/s")

        val_loss, val_acc = validate(engine, valset, engine.train_micro_batch_size_per_gpu())
        if engine.is_last_stage():
            this_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{args.epochs}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Time taken: {this_time:.2f} seconds")
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)

            # 检查是否达到目标准确率
            if val_acc >= target_accuracy and not reached_target:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Reached {target_accuracy}% accuracy at epoch {epoch + 1}, step {step}. Time taken: {elapsed_time:.2f} seconds")
                reached_target = True
                # 保存最终模型
                engine.save_checkpoint("./tmp", tag="final_model")
                return  # 提前结束训练

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
