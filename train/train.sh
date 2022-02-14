#!/bin/bash

python -u train_MultiscaleWarpingNet8_landmark.py  --dataset Flower   --display 100 --batch_size 2  --step_size 150000 --gamma 0.1  --loss CharbonnierLoss --optim Adam --lr 0.0001  --checkpoints_dir ./checkpoints/  
