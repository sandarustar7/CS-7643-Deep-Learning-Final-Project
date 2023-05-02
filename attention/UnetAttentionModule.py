from argparse import ArgumentParser
from fastmri.pl_modules import UnetModule
import torch
from UnetAttention import UnetAttention
from pytorch_lightning.tuner.tuning import Tuner

from torch import nn
from torch.nn import functional as F

from torchgeometry.losses import SSIM

class UnetAttentionModule(UnetModule):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        attention_only=False,
        unet_pretrained_path = None,
        ssim_loss = False,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.attention_only = attention_only

        print("UnetAttentionModule parameters: ")
        #print("in_chans: ", in_chans)
        #print("out_chans: ", out_chans)
        #print("chans: ", chans)
        #print("num_pool_layers: ", num_pool_layers)
        #print("drop_prob: ", drop_prob)
        print("attention_only: ", attention_only)
        print("ssim_loss: ", ssim_loss)

        self.unet = UnetAttention(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            attention_only=attention_only,
        )

        if unet_pretrained_path is not None:
            print("Using pretrained U-Net weights")
            self.load_unet_weights(torch.load(unet_pretrained_path, map_location=torch.device('cpu')))

        self.is_ssim = ssim_loss
        self.ssim_loss = SSIM(window_size=31, reduction='mean')

    def load_unet_weights(self, state_dict, check_params=True):
        missing_keys = self.unet.load_state_dict(state_dict, strict=False)
        if check_params and len(missing_keys.missing_keys) > 0:
            for k in missing_keys.missing_keys:
                assert k.split(".")[0] == 'up_attention', f"Missing key {k} in state_dict"

    def get_loss(self, output, target):
        if not self.is_ssim:
            return F.l1_loss(output, target)
        else:
            if len(output.shape) == 4:
                return self.ssim_loss(output, target)
            else:
                return self.ssim_loss(output.unsqueeze(0), target.unsqueeze(0))

    def training_step(self, batch, batch_idx):
        output = self(batch.image)

        loss = self.get_loss(output, batch.target)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": self.get_loss(output, batch.target)
        }

    def configure_optimizers(self):
        if self.attention_only:
            # Only need to optimize attention parameters
            params = []
            for name, param in self.named_parameters():
                if 'attention' in name:
                    params.append(param)
        
        else:
            # Optimize all parameters
            params = self.parameters()

        optim = torch.optim.RMSprop(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = UnetModule.add_model_specific_args(parent_parser)

        parser.add_argument(
            '--attention_only', default=False, action='store_true', help='Whether to train only the attention module'
        )

        parser.add_argument(
            '--unet_pretrained_path', type=str, default=None, help='Full path to file containing pretrained Unet weights'
        )

        parser.add_argument(
            '--ssim_loss', default=False, action='store_true', help='Whether to use SSIM loss'
        )

        return parser