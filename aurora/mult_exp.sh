#!/bin/sh

export SCRIPT_DIR="$(dirname "$(readlink -f "$0")")" #$(dirname $0)
echo $SCRIPT_DIR
export WANDB=1

########################
## SET UP FOR NODES=1 ##
########################
# ##
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=4 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=8 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=16 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=32 bash $DIR/train_pp.sh

# ##
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=4 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=6 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=8 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=16 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=32 bash $DIR/train.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=60 bash $DIR/train_pp.sh
H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=120 bash $SCRIPT_DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=16 BS=240 bash $DIR/train_pp.sh

# ##
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=4 BS=4 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=4 BS=8 bash $DIR/train_pp.sh

# ##
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=16 BS=4 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=16 BS=8 bash $DIR/train_pp.sh

# ##
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=64 BS=4 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=64 BS=8 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=64 BS=16 bash $DIR/train_pp.sh
# H_DIM=768 FFN_SIZE=3072 IMG_DIM=256 PATCH_DIM=64 BS=32 bash $DIR/train_pp.sh


########################
## SET UP FOR NODES=2 ##
########################
