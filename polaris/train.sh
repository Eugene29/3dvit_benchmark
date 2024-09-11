#!/bin/sh

module use /soft/modulefiles
module load conda
conda activate base

MONAI_DIR=$HOME
DEBUG=${DEBUG:-0}
# MONAI_DIR=$(dirname $PBS_O_WORKDIR)

## If Not DEBUG, SET-UP output folder and output.log
if [ $DEBUG -eq 0 ]; then
    DIR=$(dirname $0)
    PBS_O_WORKDIR="$DIR/h${H_DIM}_ffn${FFN_SIZE}_img${IMG_DIM}_patch${PATCH_DIM}_bs${BS}"
    mkdir -p $PBS_O_WORKDIR
    exec &> $PBS_O_WORKDIR/output.log
    FNAME=../pp_test.py
else
    PBS_O_WORKDIR=$(dirname $0)
    FNAME=pp_test.py
fi

export PYTHONPATH="$MONAI_DIR/MONAI:$PYTHONPATH"
export CUDNN_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.1.0.70/
export CPATH=$CUDNN_PATH/include:$CPATH
export CC=gcc-12
export CXX=g++-12

echo "<--------------------------- Bash Arguments --------------------------->"
echo "PBS_O_WORKDIR: $$PBS_O_WORKDIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "echo PBS_NODEFILE: $PBS_NODEFILE"
echo "cat PBS_NODEFILE: $(cat $PBS_NODEFILE)"
cd $PBS_O_WORKDIR
#export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
#export HTTPS_PROXY=http://proxy.alcf.anl.gov:3130
#export http_proxy=http://proxy.alcf.anl.gov:3128
#export https_proxy=http://proxy.alcf.anl.gov:3128
#git config --global http.proxy http://proxy.alcf.anl.gov:3128
#echo "Set HTTP_PROXY and to $HTTP_PROXY"

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
echo "MASTER NODE ${master_node} :: MASTER_ADDR ${MASTER_ADDR}"
export MASTER_PORT=29500

export NNODES=$(wc -l < $PBS_NODEFILE)
export CUDA_LAUNCH_BLOCKING=1
export NRANKS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' "\n" | wc -l | cat) ## num GPUs Visible
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
ulimit -s unlimited

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export CUDA_DEVICE_MAX_CONNECTIONS=1

## Debug Mode: rank 1
# export CUDA_VISIBLE_DEVICES=0
# export NRANKS_PER_NODE=1
# NTOTRANKS=1

## ARGS
VIT_ARGS="\
    --h_dim ${H_DIM:-768}\
    --ffn_size ${FFN_SIZE:-3072}\
    --img_dim ${IMG_DIM:-96}\
    --patch_dim ${PATCH_DIM:-16}\
    --bs ${BS:-4}\
"

## RUN CMD
python --version
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
run_cmd="mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python $FNAME $VIT_ARGS"
echo "Executing command: $run_cmd"
printf "\n\n\n\n"
echo "<------------------------ Train Script Log ------------------------->"
eval $run_cmd

#mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#mpiexec -n 1 -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#   --epochs 10 --batch_size 2 --data_root /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/multi_gpu/Covid_data  --json_path /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/datalists/tcia/dataset_split_new.json --logdir_path ./ 
cd ..
echo "Done"