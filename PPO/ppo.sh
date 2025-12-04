#!/bin/bash


# 使用DeepSpeed启动训练，并将输出重定向到日志文件
deepspeed train.py \
--num_gpus 2 \
> /media/a822/82403B14403B0E83/志明/WechatRobot/logs/ppo_train.log 2>&1