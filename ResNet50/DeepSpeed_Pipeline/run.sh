#!/bin/bash

deepspeed --hostfile=hostfile train_2.py --deepspeed_config=ds_config.json -p 2 -e 100
