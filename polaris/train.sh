#!/bin/sh

## EXAMPLE USE: 
## H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=4 bash train.sh
## DEFAULT RUN: bash train.sh

module use /soft/modulefiles
module load conda
conda activate base

## DEFAULT PARAMS
H_DIM=${H_DIM:-768}
FFN_SIZE=${FFN_SIZE:-3072}
IMG_DIM=${IMG_DIM:-96}
PATCH_DIM=${PATCH_DIM:-16}
BS=${BS:-4}

## SETUP VARS
NNODES=$(wc -l < $PBS_NODEFILE)
SEQ_LEN=$((($IMG_DIM / $PATCH_DIM) ** 3)) ## assuming cubic img and patch dim
DIR=$(dirname $0)
LOGNAME="${NNODES}node_h${H_DIM}_ffn${FFN_SIZE}_img${IMG_DIM}_patch${PATCH_DIM}_bs${BS}"
PBS_O_WORKDIR="$DIR/${NNODES}_node/$LOGNAME" ##Q. Why does everybody use PBS_O_WORKDIR?
MONAI_DIR=$(dirname $DIR)/MONAI

echo -e "Training with Hyper-parameters:
    NNODES=${NNODES}
    H_DIM=$H_DIM
    FFN_SIZE=$FFN_SIZE
    IMG_DIM=$IMG_DIM
    PATCH_DIM=$PATCH_DIM
    BS=$BS
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'all'}
Output Log and Trace File at: $PBS_O_WORKDIR/output.log \n"
mkdir -p $PBS_O_WORKDIR
exec &> $PBS_O_WORKDIR/output.log

## Curious to know more about these:
export PYTHONPATH="$MONAI_DIR:$PYTHONPATH"
export CUDNN_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.1.0.70/
export CPATH=$CUDNN_PATH/include:$CPATH
export CC=gcc-12
export CXX=g++-12

## Curious to know more about these:
export NCCL_CROSS_NIC=1 
export NCCL_COLLNET_ENABLE=1 
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH 
export FI_CXI_DISABLE_HOST_REGISTER=1 
export FI_MR_CACHE_MONITOR=userfaultfd 
export FI_CXI_DEFAULT_CQ_SIZE=131072

## Save for Aurora
#export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
#export HTTPS_PROXY=http://proxy.alcf.anl.gov:3130
#export http_proxy=http://proxy.alcf.anl.gov:3128
#export https_proxy=http://proxy.alcf.anl.gov:3128
#git config --global http.proxy http://proxy.alcf.anl.gov:3128
#echo "Set HTTP_PROXY and to $HTTP_PROXY"

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
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS=${NTOTRANKS} RANKS_PER_NODE=${NRANKS_PER_NODE}"
ulimit -s unlimited


## 1 GPU Debug Mode
# export CUDA_VISIBLE_DEVICES=0
# export NRANKS_PER_NODE=1
# NTOTRANKS=1

## PYSCRIPT ARGS
VIT_ARGS="\
    --h_dim ${H_DIM}\
    --ffn_size ${FFN_SIZE}\
    --img_dim ${IMG_DIM}\
    --patch_dim ${PATCH_DIM}\
    --bs ${BS}\
    --run_name $LOGNAME\
"

if [ -n $WANDB ]; then
    VIT_ARGS="--use_wandb $VIT_ARGS"
fi

NSYS_ARGS="--env TMPDIR=/home/eku/ --cpu-bind=numa nsys profile -o ${PBS_O_WORKDIR}/nsys/ --stats=true --show-output=true"

## RUN CMD
run_cmd="mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE $NSYS_ARGS python ../../pp_test.py $VIT_ARGS"
# $NRANKS_PER_NODE run_cmd="mpiexec -n $NTOTRANKS -ppn $NSYS_ARGS python ../../pp_test.py $VIT_ARGS"
echo "Executing command: $run_cmd"
printf "\n\n\n\n<------------------------ Train Script Log ------------------------->"
cd $PBS_O_WORKDIR
eval $run_cmd
cd ..
echo "Done"
