"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
import torch
from UnetAttentionModule import UnetAttentionModule

import gc

import ray
from ray import air, tune
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

def cli_main(args):
    pl.seed_everything(args.seed)

    if args.mode == "hyperparameter_search":
        hyperparameter_search(args)
        return

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        test_sample_rate=args.sample_rate,
        val_sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = UnetAttentionModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        attention_only=args.attention_only,
        unet_pretrained_path=args.unet_pretrained_path,
        ssim_loss=args.ssim_loss,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")
    

def hyperparameter_search(args):
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)

    assert args.accelerator == "gpu", "hyperparameter search only works for GPU"
    max_epochs = 7
    num_samples = 50

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        test_sample_rate=args.sample_rate,
        val_sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=2,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )
    
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=UnetAttentionModule, 
                lr = tune.loguniform(1e-5, 1e-1),
                weight_decay = tune.loguniform(1e-5, 1e-1),
                drop_prob = tune.uniform(0.0, 0.5),
                lr_step_size = tune.choice([30, 40, 50]),
                lr_gamma = tune.loguniform(0.8, 1e-1),
                in_chans = args.in_chans,
                out_chans = args.out_chans,
                chans = args.chans,
                num_pool_layers = args.num_pool_layers,
                attention_only = args.attention_only,
                unet_pretrained_path = args.unet_pretrained_path,
                ssim_loss = args.ssim_loss,
        )
        .trainer(
            accelerator="gpu",
            max_epochs=max_epochs,
            enable_progress_bar=False,
        )
        .fit_params(datamodule=data_module)
        .checkpointing(monitor="validation_loss", mode="min", save_top_k=1)
        .build()
    )

    run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="validation_loss",
        checkpoint_score_order="min",
        ),
    local_dir="/data/ray_results",
    )

    scheduler = ASHAScheduler(max_t = max_epochs, grace_period=2, reduction_factor=2)

    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1})

    lightning_trainer = LightningTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(metric="validation_loss", num_samples=num_samples, mode="min", scheduler=scheduler, max_concurrent_trials=6),
        run_config=RunConfig(name="tune_unet_attention", local_dir="/data/ray_results", checkpoint_config=
                             CheckpointConfig(num_to_keep=1, 
                                              checkpoint_score_attribute="validation_loss", 
                                              checkpoint_score_order="min")),
    )

    tuner = tuner.restore('/data/ray_results/tune_unet_attention', 
                          trainable=lightning_trainer, 
                          restart_errored=False, 
                          resume_unfinished=False, 
                          param_space={"lightning_config": lightning_config})
    #gc.collect()
    #results = tuner.fit()
    results = tuner.get_results()
    #tuner.restore()
    #tuner.restore('/home/sandarustar7/ray_results/tune_unet_attention')
    best_result = results.get_best_result(metric="validation_loss", mode="min")
    print(best_result)


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("fastmri_dirs.yaml")
    num_gpus = 1
    backend = "ddp_find_unused_parameters_false"
    batch_size = 1 if backend == "ddp_find_unused_parameters_false" else num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "unet" / "unet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test", "hyperparameter_search"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )

    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )

    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)

    # module config
    parser = UnetAttentionModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        accelerator="gpu",
        devices=[1, 2, 3],
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()