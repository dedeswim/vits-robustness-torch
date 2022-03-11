DATA_DIR=gs://large-ds/tensorflow_datasets/
DATASET=tfds/image_net_subset
BATCH_SIZE=256

# "Validating epsilon warm-up ablation"
# for i in {14..16} 18 19; do
#   echo "Validating model with index $i"
#   python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model xcit_nano_12_p16_224 --batch-size $BATCH_SIZE --no-normalize --checkpoint gs://robust-vits/xcit-nano-adv-training-$i/best.pth.tar --attack apgd-ce --attack-eps 4;
# done

# echo "Validating data augmentation (adversarial) ablation"
# for i in {22..37}; do
#   echo "Validating model with index $i"
#   python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model xcit_nano_12_p16_224 --batch-size $BATCH_SIZE --no-normalize --checkpoint gs://robust-vits/xcit-nano-adv-training-$i/best.pth.tar --attack apgd-ce --attack-eps 4;
# done

# echo "Validating weight decay (adversarial) ablation"
# for i in {41..47}; do
#   echo "Validating model with index $i"
#   python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model xcit_nano_12_p16_224 --batch-size $BATCH_SIZE --no-normalize --checkpoint gs://robust-vits/xcit-nano-adv-training-$i/best.pth.tar --attack apgd-ce --attack-eps 4;
# done

# echo "Validating weight decay (benign) ablation"
# for i in 2 3 4 7 10 11 13; do
#     echo "Validating model with index $i"
#     python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model xcit_nano_12_p16_224 --batch-size $BATCH_SIZE --no-normalize --checkpoint gs://robust-vits/xcit-nano-adv-training-$i/best.pth.tar;
# done

# echo "Validating DeiT"
# python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model deit_small_patch16_224 --batch-size $BATCH_SIZE --no-normalize --checkpoint gs://robust-vits/xcit-ablation-10/best.pth.tar --attack apgd-ce --attack-eps 4;

echo "Validating CaiT"
python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model cait_s12_224 --batch-size 128 --no-normalize --checkpoint gs://robust-vits/xcit-ablation-9/best.pth.tar --attack apgd-ce --attack-eps 4;

echo "Validating XCiT"
python3 validate.py $DATA_DIR --dataset $DATASET --log-wandb --model xcit_small_12_p16_224 --batch-size 128 --no-normalize --checkpoint gs://robust-vits/xcit-ablation-11/best.pth.tar --attack apgd-ce --attack-eps 4;
