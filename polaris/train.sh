#!/bin/sh

## EXAMPLE USE: H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=4 bash train.sh
## For default params: bash train.sh

module use /soft/modulefiles
module load conda
conda activate base
# . ~/venv/stable/bin/activate ## conda + fvcore

## DEFAULT_PARAMS
H_DIM=${H_DIM:-768}
FFN_SIZE=${FFN_SIZE:-3072}
IMG_DIM=${IMG_DIM:-96}
PATCH_DIM=${PATCH_DIM:-16}
BS=${BS:-4}

# DEBUG=${DEBUG:-0} ## Turns off logging
NNODES=$(wc -l < $PBS_NODEFILE)
SEQ_LEN=$((($IMG_DIM / $PATCH_DIM) ** 3)) ## assuming cubic img and patch dim

## If Not DEBUG, SET-UP output folder and output.log
# if [ $DEBUG -eq 0 ]; then
DIR=$(dirname $0)
LOGNAME="h${H_DIM}_ffn${FFN_SIZE}_img${IMG_DIM}_patch${PATCH_DIM}_bs${BS}"
PBS_O_WORKDIR="$DIR/${NNODES}_node/$LOGNAME" ##Q. Why does everybody use this?
# FNAME=../pp_test.py
MONAI_DIR=$(dirname $DIR)/MONAI
# else
#     PBS_O_WORKDIR=$(dirname $0)
#     FNAME=pp_test.py
# fi

echo -e "Training Hyper-parameters:
    NNODES=${NNODES}
    H_DIM=$H_DIM
    FFN_SIZE=$FFN_SIZE
    IMG_DIM=$IMG_DIM
    PATCH_DIM=$PATCH_DIM
    BS=$BS
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'all'}
Output Log and Trace File at: $PBS_O_WORKDIR \n"

mkdir -p $PBS_O_WORKDIR
exec &> $PBS_O_WORKDIR/output.log

export PYTHONPATH="$MONAI_DIR:$PYTHONPATH"
export CUDNN_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.1.0.70/
export CPATH=$CUDNN_PATH/include:$CPATH
export CC=gcc-12
export CXX=g++-12


cd $PBS_O_WORKDIR
## Save for Aurora
#export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
#export HTTPS_PROXY=http://proxy.alcf.anl.gov:3130
#export http_proxy=http://proxy.alcf.anl.gov:3128
#export https_proxy=http://proxy.alcf.anl.gov:3128
#git config --global http.proxy http://proxy.alcf.anl.gov:3128
#echo "Set HTTP_PROXY and to $HTTP_PROXY"

## Curious to know more about these:
export NCCL_CROSS_NIC=1 
export NCCL_COLLNET_ENABLE=1 
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH 
export FI_CXI_DISABLE_HOST_REGISTER=1 
export FI_MR_CACHE_MONITOR=userfaultfd 
export FI_CXI_DEFAULT_CQ_SIZE=131072

# set master address to the first host
master_node=$(head -1 $PBS_NODEFILE)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')
export MASTER_PORT=29500
export NNODES=$NNODES
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

## Num GPUs Visible
if [ -z $CUDA_VISIBLE_DEVICES ]; then
    export NRANKS_PER_NODE=4
else
    export NRANKS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' "\n" | wc -l | cat)
fi
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

echo "<--------------------------- Bash Arguments --------------------------->"
echo "Monai Direcotry: $MONAI_DIR"
echo "Sequence-Length (assuming cubic img and patch dim): $SEQ_LEN"
echo "PBS_O_WORKDIR: $$PBS_O_WORKDIR"
echo "PYTHONPATH: $PYTHONPATH"
# echo "echo PBS_NODEFILE: $PBS_NODEFILE"
echo "PBS_NODEFILE: $(cat $PBS_NODEFILE)"
python --version
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "MASTER NODE ${master_node} :: MASTER_ADDR ${MASTER_ADDR}"
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
ulimit -s unlimited


## 1GPU Debug Mode
# export CUDA_VISIBLE_DEVICES=0
# export NRANKS_PER_NODE=1
# NTOTRANKS=1

## PYSCRIPT ARGS
WANDB=""
if ((WANDB == 1)); then
    WANDB="--use_wandb"
fi
VIT_ARGS="\
    --h_dim ${H_DIM}\
    --ffn_size ${FFN_SIZE}\
    --img_dim ${IMG_DIM}\
    --patch_dim ${PATCH_DIM}\
    --bs ${BS}\
    --run_name $LOGNAME\
    $WANDB \
"

## RUN CMD
run_cmd="mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python ../../pp_test.py $VIT_ARGS"
echo "Executing command: $run_cmd"
printf "\n\n\n\n"
echo "<------------------------ Train Script Log ------------------------->"
eval $run_cmd

#mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#mpiexec -n 1 -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#   --epochs 10 --batch_size 2 --data_root /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/multi_gpu/Covid_data  --json_path /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/datalists/tcia/dataset_split_new.json --logdir_path ./ 
cd ..
echo "Done"
