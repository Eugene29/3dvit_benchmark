from mpi4py import MPI
import torch
import intel_extension_for_pytorch as ipex
import torch.distributed as dist
import argparse
import socket
import torch.backends.cudnn as cudnn
import os
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from torch.nn import L1Loss
from utils import collate_fn, reduce_tensor
from timm.utils import AverageMeter
import time
comm = MPI.COMM_WORLD
comm.Barrier()
world_size = comm.Get_size()
rank = comm.Get_rank()
print("RANK INFO:")
print(world_size)
print(rank)
if rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1]
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None

master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)
device_count = torch.xpu.device_count()
import torch.distributed as dist

# DEVICE_TYPE = ez.dist.get_torch_device_type()
if torch.xpu.is_available():
    torch.xpu.set_device(rank % device_count)
def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{rank%device_count}")
    else:
        return torch.device('cpu')

device  = get_default_device()
print(f"########### device :{device}")
my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)

device = 'xpu'

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
    args = parser.parse_args()
    return args


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_xpu_time_total", row_limit=10)
    p.export_chrome_trace("./trace_vit/trace_" + f"{trace_file_prefix}_step" +  str(p.step_num) + ".json")
    #print(output)
def train_epoch(epoch, data_size, model, optimizer, loss_funcs):
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    n_steps = 10
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True, profile_memory=True,
        with_stack=True, schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler) as p:
        for idx in range(n_steps):
            b_start = time.time()
            inputs = torch.randn(data_size, device=device)
            inputs_2 = torch.randn(data_size, device=device)
            gt_input = torch.randn(data_size, device=device)
       
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
        
            r_loss_t = r_loss #reduce_tensor(r_loss)
            cl_loss_t = cl_loss #reduce_tensor(cl_loss)
            total_loss_t = total_loss #reduce_tensor(total_loss)

            loss_l1_meter.update(r_loss_t.item(), inputs.size(0))
            loss_cont_meter.update(cl_loss_t.item(), inputs.size(0))
            loss_meter.update(total_loss_t.item(), inputs.size(0))
            bt = time.time() - b_start
            batch_time.update(bt)
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.xpu.max_memory_allocated() / (1024.0 * 1024.0)
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
        print(f"Avg Time per step for epoch:{epoch}= {batch_time.avg} \t"
              f"Thoughput in samples/sec = {(data_size[0]*n_steps)/batch_time.sum} \t"
              f"mem used : {memory_used:.0f}MB")
trace_file_prefix = ""
def main(args):
    device = 'xpu'
    img_size = (96, 96, 96)
    patch_size = (16, 16, 16)
    n_channels = 1
    bs = 32
    data_size = (bs,n_channels,img_size[0],img_size[1],img_size[2])
    print(data_size)
    global trace_file_prefix 
    trace_file_prefix = f"bs{bs}_ch{n_channels}_img{img_size[0]}_patch{patch_size[0]}"
    print(f"trace_file: {trace_file_prefix}")
    model = ViTAutoEnc(
        in_channels=n_channels,
        img_size=img_size,
        patch_size=patch_size,
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
    ).xpu()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank%device_count], broadcast_buffers=False, find_unused_parameters=True
    )
    model_without_ddp = model.module
    #data_size = (2,1,96,96,96)
    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_funcs = [recon_loss, contrastive_loss]
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))
    print("Training Begins ...")
    
    for epoch in range(args.epochs):
        train_epoch(epoch, data_size, model, optimizer, loss_funcs)

if __name__ == "__main__":
    args = parse_option()
    import oneccl_bindings_for_pytorch
    torch.distributed.init_process_group(backend="ccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = args.seed + dist.get_rank()
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
