#!/bin/sh

#module use /soft/modulefiles
#module load conda
#conda activate base
# module load python
# source $IDPROOT/etc/profile.d/conda.sh ## or 
# module load use /home/jmitche1/anl_release/aurora/2024/q3 ; module load frameworks_2024_8.lua

## EXAMPLE USE: 
## H_DIM=768 FFN_SIZE=3072 IMG_DIM=96 PATCH_DIM=4 BS=4 bash train.sh
## DEFAULT RUN: bash train.sh

module load frameworks/2024.1
source ~/multi_gpu/venv/vit3d/bin/activate #~/venv/stable/bin/activate ## timm

## DEFAULT_PARAMS
H_DIM=${H_DIM:-768}
FFN_SIZE=${FFN_SIZE:-3072}
IMG_DIM=${IMG_DIM:-96}
PATCH_DIM=${PATCH_DIM:-16}

echo "BS: $BS"
## SETUP VARS
echo "SCRIPT_DIR: $SCRIPT_DIR"
NNODES=$(wc -l < $PBS_NODEFILE)
SEQ_LEN=$((($IMG_DIM / $PATCH_DIM) ** 3)) ## assuming cubic img and patch dim

LOGNAME="h${H_DIM}_ffn${FFN_SIZE}_img${IMG_DIM}_patch${PATCH_DIM}_bs${BS}_$(date +"%Y-%m-%d-%H-%M-%S")"
PBS_O_WORKDIR="$SCRIPT_DIR/${NNODES}_node/$LOGNAME" ##Q. Why does everybody use PBS_O_WORKDIR?
MONAI_DIR=$(realpath $(dirname $SCRIPT_DIR))/MONAI #$(dirname $DIR)/MONAI
echo "logname: $LOGNAME"
echo "PWD: $PWD"
echo "PBS: $PBS_O_WORKDIR"
echo "MONAI DIR : $MONAI_DIR"
echo -e "Training with Hyper-parameters:
    NNODES=${NNODES}
    H_DIM=$H_DIM
    FFN_SIZE=$FFN_SIZE
    IMG_DIM=$IMG_DIM
    PATCH_DIM=$PATCH_DIM
    BS=$BS
Output Log and Trace File at: $PBS_O_WORKDIR \n"
mkdir -p $PBS_O_WORKDIR
exec &> $PBS_O_WORKDIR/output.log

## Curious to know more about these:
export PYTHONPATH="$MONAI_DIR:$PYTHONPATH"
export FI_CXI_DISABLE_HOST_REGISTER=1 
export FI_MR_CACHE_MONITOR=userfaultfd 
export FI_CXI_DEFAULT_CQ_SIZE=131072


git config --global http.proxy http://proxy.alcf.anl.gov:3128
echo "Set HTTP_PROXY and to $HTTP_PROXY"

# set master address to the first host
master_node=$(head -1 $PBS_NODEFILE)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')
export MASTER_PORT=29500
export NNODES=$NNODES

## Num GPUs Visible
if [ -z $CUDA_VISIBLE_DEVICES ]; then
    export NRANKS_PER_NODE=6
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


## PYSCRIPT ARGS
if [ -n WANDB ]; then
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
run_cmd="mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python $SCRIPT_DIR/pp_test.py $VIT_ARGS"
echo "Executing command: $run_cmd"
printf "\n\n\n\n"
echo "<------------------------ Train Script Log ------------------------->"
cd $PBS_O_WORKDIR
eval $run_cmd
cd ..
echo "Done"
