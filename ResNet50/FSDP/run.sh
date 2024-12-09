export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=1 train_pytorch.py --num-epochs 100 --batch-size 1024
