# Standard library
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from neural_lam import constants, utils
from neural_lam.models.mask_efm import GraphEFM_mask
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import ERA5toCERRA, ERA5toCERRA2
import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,5"

MODELS = {
    "graphcast": GraphCast,
    "graph_efm": GraphEFM_mask,
}


def get_args():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    # General options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset_cerra",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_era5",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_efm",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical_HighLow_res",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--graph_conditioner",
        type=str,
        default="hierarchical_HighLow_res",
        help="Graph to load and use in graph-based model, this on is for conditioning"
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 4)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in prior GNN (default: 2)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--vertical_propnets",
        type=int,
        default=0,
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 0 (no))",
    )
    # Training options
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.25,
        help="Masking ratio of original grid",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-lam",
        help="Wandb project name",
    )
    parser.add_argument(
        "--kl_term",  
        type=int,
        default=1,
        help="KL term in loss function",
    )
    parser.add_argument(
        "--kl_beta",  
        type=float,
        default=0.01,
        help="KL beta term in loss function",
    )
    return parser.parse_args()

def main(args):
    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)
    N_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    print(f"Using {N_GPUS} GPUs")

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        ERA5toCERRA2(
            args.dataset_cerra,
            args.dataset_era5,
            split="train",
            subset=bool(args.subset_ds),
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    
    val_loader = torch.utils.data.DataLoader(
        ERA5toCERRA2(
            args.dataset_cerra,
            args.dataset_era5,
            split="val",
            subset=bool(args.subset_ds),
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        #torch.set_float32_matmul_precision(
        #    "high"
        #)  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
    )
    
    if args.wandb_project is not None:
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project, name=run_name, config=args
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir="DebugLogs/", name=run_name
        )  # or CSVLogger


    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        #devices=2,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        #profiler="simple",
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0 and isinstance(logger, pl.loggers.WandbLogger):
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                ERA5toCERRA2(
                    args.dataset_cerra,
                    args.dataset_era5,
                    split="test",#TODO: Change to val
                    subset=False,
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            
        )


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  
  

if __name__ == '__main__':
    
    args = get_args()

    if args.config is not None:

        with open(str(args.config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)

    main(args)
