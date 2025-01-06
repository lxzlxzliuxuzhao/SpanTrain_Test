#!/bin/bash

deepspeed train_2.py --deepspeed_config=ds_config.json -p 1 -e 100
