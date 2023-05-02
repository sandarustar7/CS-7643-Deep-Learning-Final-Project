

import argparse
# from fastmri.pl_modules import UnetCustomLossModule, UnetModule
from modules.unet import UnetCustomLossModule

import torch

import fastmri

def dump_state_dictionary(args):
    model = UnetCustomLossModule.load_from_checkpoint(args.checkpoint_path, in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
    state_dict = model.unet.state_dict()
    print(len(state_dict))
    torch.save(state_dict, args.output_path)
    print('Saved state dictionary to ', args.output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    return parser.parse_args()

def __main__():
    args = parse_args()
    dump_state_dictionary(args)

if __name__ == '__main__':
    __main__()
