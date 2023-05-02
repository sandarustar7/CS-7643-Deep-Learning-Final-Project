

import argparse

import torch
from UnetAttentionModule import UnetAttentionModule
import fastmri

def dump_state_dictionary(args):
    model = UnetAttentionModule.load_from_checkpoint(args.checkpoint_path)
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
