from pathlib import Path
import torch
import pickle
import argparse
import logging
import torch.distributed as dist
from config import MyParser
from steps import trainer
import os

def set_torch_config():

    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEFAULT_DEVICE = "mps"
    else:
        DEFAULT_DEVICE = "cpu"

    if DEFAULT_DEVICE == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    logging.debug(f"default Torch device: {DEFAULT_DEVICE}")
    torch.set_default_device(DEFAULT_DEVICE)

    return DEFAULT_DEVICE

if __name__ == "__main__":

    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    device = set_torch_config()
    print('starting...')

    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    logging.info(args)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"exp_dir: {str(exp_dir)}")

    if args.resume:
        resume = args.resume
        assert(bool(args.exp_dir))
        with open("%s/args.pkl" % args.exp_dir, "rb") as f:
            old_args = pickle.load(f)
        new_args = vars(args)
        old_args = vars(old_args)
        for key in new_args:
            if key not in old_args or old_args[key] != new_args[key]:
                old_args[key] = new_args[key]
        args = argparse.Namespace(**old_args)
        args.resume = resume
    else:
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)
            
    # Disable libuv because PyTorch for Windows isn't built with support
    os.environ["USE_LIBUV"] = "0"

    # initialize process group
    # Windows users may have to use "gloo" instead of "nccl" as backend
    # nccl: NVIDIA Collective Communication Library
    # gloo: Facebook Collective Communication Library
    if device == "cuda":
        if torch.cuda.nccl.is_available(torch.randn(1).cuda()):
            logging.info('Using cuda + nccl')
            dist.init_process_group("nccl", init_method='env://')
        else:
            logging.info('nccl not available. Using gloo with device '+str(device))
            dist.init_process_group("gloo", init_method='env://')
    else:
        logging.info('Using gloo with device '+str(device))
        dist.init_process_group("gloo", init_method='env://')

    # Check what version of PyTorch is installed
    logging.info(torch.__version__)

    # Check the current CUDA version being used
    logging.info("CUDA Version: ", torch.version.cuda)

    # Check if CUDA is available and if so, print the device name
    logging.info("Device name:", torch.cuda.get_device_properties("cuda").name)

    # Check if FlashAttention is available
    logging.info("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # torch.set_device(rank)
    my_trainer = trainer.Trainer(args, world_size, rank)
    # set MPS as fallback device PYTORCH_ENABLE_MPS_FALLBACK=1

    my_trainer.train()