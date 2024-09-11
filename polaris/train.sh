#!/bin/sh

#module use /soft/modulefiles
#module load conda
#conda activate base


cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

export CUDNN_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.1.0.70/
export CPATH=$CUDNN_PATH/include:$CPATH
export CC=gcc-12
export CXX=g++-12
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

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NNODES=$(wc -l < $PBS_NODEFILE)
export NRANKS_PER_NODE=4
export CUDA_LAUNCH_BLOCKING=1
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
cat $PBS_NODEFILE

ulimit -s unlimited

cd $PBS_O_WORKDIR
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NNODES=$NNODES
export NRANKS_PER_NODE=4


python --version
python -c "import torch; print(torch.__version__)"
echo "Executing command mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE *.py "
mpiexec -n 1 -ppn $NRANKS_PER_NODE python pp_test.py #\
#mpiexec -n $NTOTRANKS -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#mpiexec -n 1 -ppn $NRANKS_PER_NODE python mgpu_ssl_train_random_profile.py \
#   --epochs 10 --batch_size 2 --data_root /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/multi_gpu/Covid_data  --json_path /eagle/datascience/vsastry/Vit_Pipeline/tutorials/self_supervised_pretraining/vit_unetr_ssl/datalists/tcia/dataset_split_new.json --logdir_path ./ 
echo "Done"
