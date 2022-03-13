DATA_DIR=gs://large-ds/tensorflow_datasets/
BATCH_SIZE=256

DATASET=tfds/caltech101
python3 validate.py $DATA_DIR --dataset $DATASET \
    --log-freq 1 --model resnet50 --batch-size 256 \
    --checkpoint gs://robust-vits/xcit-adv-finetuning-hi-res-5/best.pth.tar \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --split test \
    --normalize-model --num-classes 102 \
    --attack autoattack --num-examples 5000 --log-wandb --attack-eps 8

echo "Validating XCiT-S on Caltech-101"
python3 validate.py $DATA_DIR --dataset $DATASET \
    --log-freq 1 --model xcit_small_12_p16_224 --batch-size 128 \
    --checkpoint gs://robust-vits/xcit-adv-finetuning-hi-res-3/best.pth.tar \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --split test \
    --normalize-model --num-classes 102 \
    --attack autoattack --num-examples 5000 --log-wandb --attack-eps 8

DATASET=tfds/oxford_flowers102
echo "Validating ResNet-50 on Oxford Flowers"
python3 validate.py $DATA_DIR --dataset $DATASET \
    --log-freq 1 --model resnet50 --batch-size 256 \
    --checkpoint gs://robust-vits/xcit-adv-finetuning-hi-res-4/best.pth.tar \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --split test \
    --normalize-model --num-classes 102 \
    --attack autoattack --num-examples 5000 --log-wandb --attack-eps 8

echo "Validating XCiT-S on Oxford Flowers"
python3 validate.py $DATA_DIR --dataset $DATASET \
    --log-freq 1 --model xcit_small_12_p16_224 --batch-size 128 \
    --checkpoint gs://robust-vits/xcit-adv-finetuning-hi-res-2/best.pth.tar \
    --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --split test \
    --normalize-model --num-classes 102 \
    --attack autoattack --num-examples 5000 --log-wandb --attack-eps 8