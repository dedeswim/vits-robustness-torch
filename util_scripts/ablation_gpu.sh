#!/bin/bash

# ->->->->->-> select key variables based on princeton or epfl <-<-<-<-<-<-
cluster="pn" # switch between pn/ep

if [ $cluster == "pn" ] 
then
    ROOT_DIR="/shadowdata/vvikash/spring22/edoardo/vits-robustness-torch"
    DATADIR="${ROOT_DIR}/datasets/"
    CKPTDIR="${ROOT_DIR}/checkpoints/"
    SAVEDIR="${ROOT_DIR}/cifar_finetune/"
elif [ $cluster == "ep" ] 
then
    DATADIR="~/torch_data/"
    CKPTDIR="gs://robust-vits/xcit-adv-pretraining-3/best.pth.tar"
    SAVEDIR="./output/tmp/"
else
    echo "cluster not supported, terminating program!"
    exit
fi


# We will control which experiments to run using SET variable.
SET=1
echo "Running set ${SET} experiments"


# ablation with trades (cifar10) where lr: (5e-6, 1e-5, 5e-5, 1e-4), beta: (6, 12), wd: (0.05, 0.5, 1.0). Note that all ablations are down with batch=size=2*64
if [ $SET -eq 1 ] 
then
    DATASET="--dataset torch/cifar10 --num-classes 10"
    NORMALIZATION="--mean 0.4914 0.4822 0.4465 --std 0.2471 0.2435 0.2616 --normalize-model"
    ATTACK="--attack-steps 10 --attack-eps 8 --adv-training trades"
    SETUP="--config configs/xcit-adv-finetuning.yaml --finetune ${CKPTDIR}/xcit_s_eps_8.pth.tar --output ${SAVEDIR} --sync-bn --log-wandb"
    
    for wd in 0.05 0.5 1.0
    do 
        HYPERPARAMS="--epochs 20 --color-jitter 0.0 --cutmix 0.0 --reprob 0.20 --weight-decay ${wd} --smoothing 0.0 
            --cooldown-epoch 2"
        beta=6.0
        lr=0.00005
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6712 train.py \
            $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr --trades-beta ${beta} \
            --experiment "xcit-adv-finetuning-gpu_cifar10_ablations_only_randErasing" &
        sleep 1
        lr=0.0001
        CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6713 train.py \
            $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr --trades-beta ${beta} \
            --experiment "xcit-adv-finetuning-gpu_cifar10_ablations_only_randErasing" &
        sleep 1
        beta=12.0
        lr=0.00005
        CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6714 train.py \
            $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr --trades-beta ${beta} \
            --experiment "xcit-adv-finetuning-gpu_cifar10_ablations_only_randErasing" &
        sleep 1
        lr=0.0001
        CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6715 train.py \
            $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr --trades-beta ${beta} \
            --experiment "xcit-adv-finetuning-gpu_cifar10_ablations_only_randErasing" ;
        wait;
    done
fi

# ablation with trades (cifar100) where lr: (5e-6, 1e-5, 5e-5, 1e-4), beta: (6, 12), wd: (0.05, 0.5, 1.0). Note that all ablations are down with batch=size=2*64
if [ $SET -eq 2 ] 
then
    DATASET="--dataset torch/cifar100 --num-classes 100"
    NORMALIZATION="--mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --normalize-model"
    ATTACK="--attack-steps 10 --attack-eps 8 --adv-training trades"
    SETUP="--config configs/xcit-adv-finetuning.yaml --finetune ${CKPTDIR}/xcit_s_eps_8.pth.tar --output ${SAVEDIR} --sync-bn --log-wandb"
    
    for beta in 6 12
    do 
        for wd in 0.5 1.0
        do 
            HYPERPARAMS="--epochs 20 --color-jitter 0.0 --cutmix 0.0 --reprob 0.20 --weight-decay ${wd} --smoothing 0.0 \
                --cooldown-epoch 2  --trades-beta ${beta}"
            echo $SETUP
            lr=0.00005
            CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6713 train.py \
                $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr \
                --experiment "xcit-adv-finetuning-gpu_cifar100_ablations_only_randErasing" &
            sleep 1
            lr=0.0001
            CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6714 train.py \
                $DATADIR $DATASET $SETUP $NORMALIZATION $ATTACK $HYPERPARAMS --batch-size 64 --lr $lr \
                --experiment "xcit-adv-finetuning-gpu_cifar100_ablations_only_randErasing" ;
            wait;
        done
    done
fi









     
