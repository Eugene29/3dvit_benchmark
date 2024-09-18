from mpi4py import MPI
import torch
import torch.distributed as dist
import argparse
import torch.backends.cudnn as cudnn
import os
import torch.distributed
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from torch.nn import L1Loss
from utils import collate_fn, reduce_tensor
from timm.utils import AverageMeter
import numpy as np
import time
from deepspeed.profiling.flops_profiler import FlopsProfiler

comm = MPI.COMM_WORLD
comm.Barrier()
world_size = comm.Get_size()
rank = comm.Get_rank()
def print_rank_0(msg: str):
    if rank == 0:
        print(msg)
device_count = torch.cuda.device_count()

if torch.cuda.is_available():
    torch.cuda.set_device(rank % device_count)

my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)
device = 'cuda'

def parse_option():
    parser = argparse.ArgumentParser("ViT Self-Supervised Learning", add_help=False)

    # Set Paths for running SSL training
    #parser.add_argument("--data_root", default="/workspace/datasets/tcia/", type=str, help="path to data root")
    #parser.add_argument(
    #    "--json_path",
    #    default="./datalists/tcia/dataset_split.json",
    #    type=str,
    #    help="Json file path for list of data samples",
    #)
    #parser.add_argument("--logdir_path", default="/to/be/defined", type=str, help="output log directory")
    #parser.add_argument(
    #    "--output",
    #    default="output",
    #    type=str,
    #    metavar="PATH",
    #    help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    #)

    # DL Training Hyper-parameters
    parser.add_argument("--epochs", default=5, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size for single GPU")
    parser.add_argument("--base_lr", default=5e-4, type=float, help="base learning rate")
    parser.add_argument("--seed", default=19, type=int, help="seed")
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    
    ## EK
    parser.add_argument("--h_dim", type=int, default=768, help="patch embedding size likely?")
    parser.add_argument("--ffn_size", type=int, default=3072, help="feedforward neural network size")
    parser.add_argument("--img_dim", type=int, default=96, help="image size (cubed)")
    parser.add_argument("--patch_dim", type=int, default=16, help="patch size (cubed)")
    parser.add_argument("--bs", type=int, default=4, help="global batch size")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()
    return args


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    os.makedirs("./trace_vit/", exist_ok=True)
    if rank == 0: ## avoid race condition error
        p.export_chrome_trace("./trace_vit/trace_" + f"{trace_file_prefix}_step" +  str(p.step_num) + ".json")
    #print(output)

def train_epoch(epoch, data_size, model, optimizer, loss_funcs, samples_per_secs, tflops_lst, prof):
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    n_steps = 10
    profile_step = 5

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True,
        with_stack=True, schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler) as p:
        for idx in range(n_steps):
            if idx == profile_step:
                prof.start_profile()
            b_start = time.time()
            inputs = torch.randn(data_size, device=device)
            inputs_2 = torch.randn(data_size, device=device)
            gt_input = torch.randn(data_size, device=device) ##Q. ?
       
            optimizer.zero_grad()
            with record_function("model_inference"):
                outputs_v1, hidden_v1 = model(inputs)
                outputs_v2, hidden_v2 = model(inputs_2)
        
                flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
                flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)
        
                r_loss = loss_funcs[0](outputs_v1, gt_input)
                cl_loss = loss_funcs[1](flat_out_v1, flat_out_v2)

                # Adjust the CL loss by Recon Loss
                total_loss = r_loss + cl_loss * r_loss
            with record_function("backward_pass"):
                total_loss.backward()
            optimizer.step()
            p.step()
        
            # end profiling and print output
            if idx == profile_step: # if using multi nodes, check global_rank == 0 as well
                prof.stop_profile()
                total_flops = prof.get_total_flops()
                macs = prof.get_total_macs()
                params = prof.get_total_params()
                prof.print_model_profile(profile_step=profile_step, detailed=False)
                flops = total_flops / prof.get_total_duration() / (1000**4)
                prof.end_profile()
                print_rank_0(f"TFlops: {flops}")
                tflops_lst.append(flops)

            r_loss_t = reduce_tensor(r_loss)
            cl_loss_t = reduce_tensor(cl_loss)
            total_loss_t = reduce_tensor(total_loss)
            # r_loss_t = r_loss #reduce_tensor(r_loss)
            # cl_loss_t = cl_loss #reduce_tensor(cl_loss)
            # total_loss_t = total_loss #reduce_tensor(total_loss)

            loss_l1_meter.update(r_loss_t.item(), inputs.size(0))
            loss_cont_meter.update(cl_loss_t.item(), inputs.size(0))
            loss_meter.update(total_loss_t.item(), inputs.size(0))
            bt = time.time() - b_start
            batch_time.update(bt)
            lr = optimizer.param_groups[0]["lr"]
            """
            etas = batch_time.avg * (num_steps - idx)
            print(
                f"Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t"
                f"loss_MM {loss_cont_meter.val:.4f} ({loss_cont_meter.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
            """
        # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        samples_per_sec = (data_size[0]*n_steps)/batch_time.sum
        samples_per_secs.append(samples_per_sec)
        # print_rank_0(f"Avg Time per step for epoch:{epoch}= {batch_time.avg} \t"
        #       f"Thoughput in samples/sec = {samples_per_sec} \t"
        #       f"mem used : {memory_used:.0f}MiB")
trace_file_prefix = ""

def main(args):
    bs, img_dim, patch_dim, h_dim, ffn_size = args.bs, args.img_dim, args.patch_dim, args.h_dim, args.ffn_size
    assert args.bs % world_size == 0, "Batch size not divisible by DP RANK"
    bs = bs // world_size
    n_channels = 1
    img_size = (img_dim, img_dim, img_dim) ## Cubed
    patch_size = (patch_dim, patch_dim, patch_dim) ## Cubed
    data_size = (bs,n_channels,img_size[0],img_size[1],img_size[2])
    global trace_file_prefix
    trace_file_prefix = f"bs{bs}_ch{n_channels}_img{img_size[0]}_patch{patch_size[0]}"

    print_rank_0(f"\n\n\n\n<------------------------ Result ------------------------->")
    print_rank_0(f"world_size: {world_size}")
    print_rank_0(f"pyscript arguments: {vars(args)}")
    print_rank_0(f"data size: {data_size}")
    print_rank_0(f"trace_file_prefix: {trace_file_prefix}")
    print(f"RANK INFO: {rank}")

    ## Init Model
    model = ViTAutoEnc(
        in_channels=n_channels,
        img_size=img_size,
        patch_size=patch_size,
        # pos_embed="conv", ##Q. This argument is not used, deprecated for proj_type?
        proj_type="conv",
        hidden_size=h_dim,
        mlp_dim=ffn_size,
    ).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank%device_count], broadcast_buffers=False, find_unused_parameters=True
    )
    # model_without_ddp = model.module
    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_funcs = [recon_loss, contrastive_loss]
    samples_per_secs, tflops_lst = [], []
    # loss_l1_meter = AverageMeter()
    # loss_cont_meter = AverageMeter()
    # loss_meter = AverageMeter()
    # batch_time = AverageMeter()
    

    ## Train
    prof = FlopsProfiler(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0("number of params: {}".format(n_parameters))
    print_rank_0("Training Begins ...")

    for epoch in range(args.epochs):
        train_epoch(epoch, data_size, model, optimizer, loss_funcs, samples_per_secs, tflops_lst, prof)

    ## Log
    avg_samples_per_sec = sum(samples_per_secs[1:]) / len(samples_per_secs[1:])
    max_memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    avg_tflops = sum(tflops_lst[1:]) / len(tflops_lst[1:])
    result = {"avg_samples_per_sec (per GPU)": avg_samples_per_sec, "max_memory (GiB)": max_memory_used, "avg TFlops (per GPU)": avg_tflops}
    if args.use_wandb and rank==0:
        wandb.log(result)
    print_rank_0(result)


if __name__ == "__main__":
    args = parse_option()

    if args.use_wandb and rank==0:
        import wandb
        if args.run_name:
            wandb.init(project="3dvit", name=args.run_name)
        else:
            wandb.init(project="3dvit")

    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = args.seed + dist.get_rank() ##Q. Why do we do this? 
    if args.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)
    cudnn.benchmark = True
    """
    if dist.get_rank() == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.logdir_path):
            os.makedirs(args.logdir_path, exist_ok=True)

    if dist.get_rank() == 0:
        log_writer = TensorboardLogger(log_dir=args.logdir_path)
    """
    main(args)
