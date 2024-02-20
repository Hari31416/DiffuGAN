from .gan import FCNGenerator, FCNDiscriminator, FCNGAN
from DiffuGAN.utils import (
    add_dataset_args,
    add_wandb_args,
    create_simple_logger,
    create_wandb_logger,
    ImageDataset,
)

import argparse
import yaml
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(file_dir, "gan_config.yaml")
logger = create_simple_logger(__name__)


def load_config(file_path: str) -> dict[str:any]:
    """Loads the configuration from a file."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_default_config() -> dict[str:any]:
    """Loads the default configuration."""
    return load_config(DEFAULT_CONFIG_PATH)


def update_config(
    default_config: dict[str:any], new_config: dict[str:any]
) -> dict[str:any]:
    """Updates the default configuration with the new configuration."""
    for key, value in new_config.items():
        if key in default_config:
            if isinstance(value, dict):
                update_config(default_config[key], value)
            else:
                if value is not None:
                    default_config[key] = value
    return default_config


def create_configs(args: argparse.ArgumentParser) -> dict[str : dict[str:any]]:
    """Creats the configuration from the arguments. Th function returns a dictionary of dictionaries. The keys are:

    - dataset_config: The configuration for the dataset.
    - wandb_config: The configuration for wandb.
    - generator_config: The configuration for the generator.
    - discriminator_config: The configuration for the discriminator.
    - gan_config: The configuration for the GAN.
    """
    # dataset_config
    dataset_config = {
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "root": args.root,
        "num_workers": args.num_workers,
        "scale_type": args.scale_type,
        "shuffle": args.shuffle,
        "train": args.split_train,
    }

    # wandb_config
    if args.wandb:
        wandb_config = {
            "project": args.project,
            "name": args.run,
            "notes": args.notes,
            "tags": args.tags,
            "group": args.group,
        }
    else:
        wandb_config = {}

    # generator_config
    generator_config = {
        "latent_dimension": args.latent_dimension,
        "layer_sizes": list(map(int, args.g_layer_sizes)),
        "activation": args.g_activation,
        "activation_kwargs": {"negative_slope": args.g_activation_negative_slope},
        "final_activation": args.g_final_activation,
    }

    # discriminator_config
    d_config = {
        "layer_sizes": list(map(int, args.d_layer_sizes)),
        "dropout_rates": list(map(float, args.d_dropout_rates)),
        "activation": args.d_activation,
        "activation_kwargs": {"negative_slope": args.d_activation_negative_slope},
    }

    # gan_config
    gan_config = {
        "k": args.k,
        "generator_optimizer": args.g_optimizer,
        "discriminator_optimizer": args.d_optimizer,
        "generator_optimizer_kwargs": {"lr": args.g_lr, "betas": args.betas},
        "discriminator_optimizer_kwargs": {"lr": args.d_lr, "betas": args.betas},
        "epochs": args.epochs,
        "max_iteration_per_epoch": args.max_iteration_per_epoch,
        "log_interval": args.log_interval,
        "generator_loss_to_use": args.generator_loss_to_use,
        "max_step_for_og_loss": args.max_step_for_og_loss,
        "image_plot_interval": args.image_plot_interval,
        "image_save_path": args.image_save_path,
        "image_save_interval": args.image_save_interval,
        "model_save_path": args.model_save_path,
    }
    config = {
        "dataset_config": dataset_config,
        "wandb_config": wandb_config,
        "generator_config": generator_config,
        "discriminator_config": d_config,
        "gan_config": gan_config,
    }
    logger.info("Configuration created. Here is the configuration:\n")
    logger.info(f"Dataset config: {dataset_config}\n")
    logger.info(f"Wandb config: {wandb_config}\n")
    logger.info(f"Generator config: {generator_config}\n")
    logger.info(f"Discriminator config: {d_config}\n")
    logger.info(f"GAN config: {gan_config}\n")
    return config


def add_generator_args(
    args: argparse.ArgumentParser, default_arguments: dict[str:any]
) -> argparse.ArgumentParser:
    """Adds the generator arguments to the argument parser."""
    args.add_argument(
        "--latent-dimension",
        type=int,
        default=default_arguments.get("latent_dimension", 100),
        help="The latent dimension.",
    )
    args.add_argument(
        "--g-layer-sizes",
        type=str,
        nargs="+",
        default=default_arguments.get("layer_sizes", [128, 256, 512]),
        help="The number of neurons in the layers.",
    )
    args.add_argument(
        "--g-activation",
        type=str,
        default=default_arguments["activation"].get("name", "relu"),
        help="The activation function to use for the generator.",
    )
    args.add_argument(
        "--g-activation-negative-slope",
        type=float,
        default=default_arguments["activation"]["params"].get("negative_slope", 1e-2),
        help="The negative slope for the LeakyReLU activation. To be used only if the activation is LeakyReLU.",
    )
    args.add_argument(
        "--g-final-activation",
        type=str,
        default=default_arguments.get("final_activation", "tanh"),
        help="The final activation function to use for the generator.",
        choices=["tanh", "sigmoid"],
    )
    return args


def add_discriminator_args(
    args: argparse.ArgumentParser, default_arguments: dict[str:any]
) -> argparse.ArgumentParser:
    """Adds the generator arguments to the argument parser."""
    args.add_argument(
        "--d-layer-sizes",
        type=str,
        nargs="+",
        default=default_arguments.get("layer_sizes", [128, 256, 512]),
        help="The number of neurons in the layers.",
    )
    args.add_argument(
        "--d-activation",
        type=str,
        default=default_arguments["activation"].get("name", "relu"),
        help="The activation function to use.",
    )
    args.add_argument(
        "--d-activation-negative-slope",
        type=float,
        default=default_arguments["activation"]["params"].get("negative_slope", 1e-2),
        help="The negative slope for the LeakyReLU activation. To be used only if the activation is LeakyReLU.",
    )
    args.add_argument(
        "--d-dropout-rates",
        type=float,
        nargs="+",
        default=default_arguments.get("dropout_rates", [0.3, 0.3]),
    )
    return args


def add_gan_args(
    args: argparse.ArgumentParser, default_arguments: dict[str:any]
) -> argparse.ArgumentParser:
    """Adds the GAN arguments to the argument parser."""
    args.add_argument(
        "--k",
        "-k",
        type=int,
        default=default_arguments.get("k", 1),
        help="The number of steps to train the discriminator.",
    )
    args.add_argument(
        "--g-lr",
        type=float,
        default=default_arguments["generator_optimizer"]["params"].get("lr", 0.0001),
        help="The learning rate for the generator.",
    )
    args.add_argument(
        "--g-optimizer",
        type=str,
        default=default_arguments["generator_optimizer"].get("name", "adam"),
        help="The optimizer to use for the generator.",
    )
    args.add_argument(
        "--d-lr",
        type=float,
        default=default_arguments["discriminator_optimizer"]["params"].get(
            "lr", 0.0001
        ),
        help="The learning rate for the discriminator.",
    )
    args.add_argument(
        "--d-optimizer",
        type=str,
        default=default_arguments["discriminator_optimizer"].get("name", "adam"),
        help="The optimizer to use for the discriminator.",
    )
    args.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=default_arguments["generator_optimizer"]["params"].get(
            "betas", [0.5, 0.999]
        ),
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=default_arguments.get("epochs", 10),
        help="The number of epochs.",
    )
    args.add_argument(
        "--max-iteration-per-epoch",
        type=int,
        default=default_arguments.get("max_iteration_per_epoch", 10),
        help="The maximum number of iterations per epoch.",
    )
    args.add_argument(
        "--log-interval",
        type=int,
        default=default_arguments.get("log_interval", 100),
        help="The number of iterations after which to log the loss.",
    )
    args.add_argument(
        "--image-plot-interval",
        type=int,
        default=default_arguments.get("image_plot_interval", 100),
        help="The number of iterations after which to plot the sample images.",
    )
    args.add_argument(
        "--generator-loss-to-use",
        type=str,
        default=default_arguments.get("generator_loss_to_use", "bce"),
        choices=["bce", "og", "og_bce"],
        help="The loss to use for the generator. See the documentation for more details.",
    )
    args.add_argument(
        "--max-step-for-og-loss",
        type=int,
        default=default_arguments.get("max_step_for_og_loss", 1000),
        help="The maximum step for og loss if og_bce is used.",
    )
    args.add_argument(
        "--image-save-path",
        type=str,
        default=default_arguments.get("image_save_path", None),
        help="The path to save the images.",
    )
    args.add_argument(
        "--image-save-interval",
        type=int,
        default=default_arguments.get("image_save_interval", 100),
        help="The number of iterations after which to save the images.",
    )
    args.add_argument(
        "--model-save-path",
        type=str,
        default=default_arguments.get("model_save_path", None),
        help="The path to save the final model.",
    )
    return args


def arg_parse() -> argparse.Namespace:
    """Parses the arguments."""
    default_config = load_default_config()
    dataset_config = default_config["dataset_config"]
    wandb_config = default_config["wandb_config"]
    generator_config = default_config["generator_config"]
    d_config = default_config["discriminator_config"]
    gan_config = default_config["gan_config"]
    args = argparse.ArgumentParser(add_help=True)
    args.add_argument(
        "--f",
        type=str,
        default="",
        help="The configuration file to use. If not provided, the default configuration will be used.",
    )
    args = add_dataset_args(args, dataset_config)
    args = add_wandb_args(args, wandb_config)
    args = add_generator_args(args, generator_config)
    args = add_discriminator_args(args, d_config)
    args = add_gan_args(args, gan_config)
    return args.parse_args()


def main(args: argparse.Namespace) -> None:
    """The main function. The function creates the configuration and trains the GAN."""
    if args.f:
        # if a file is provided, load the configuration from the file
        logger.info(f"Loading the configuration from {args.f}")
        s_config = load_config(args.f)
        config = update_config(load_default_config(), s_config)
    else:
        # create the configuration from the arguments
        logger.info("Creating the configuration from the arguments.")
        config = create_configs(args)
    logger.debug(config)
    # return
    # load the dataset
    dataset = ImageDataset().load_dataset(**config["dataset_config"])
    # load a batch to get the image shape
    batch = next(iter(dataset))
    image_shape = batch[0].shape[1:]
    # convert image shape to tuple
    image_shape = tuple(image_shape)
    logger.info(f"Dataset loaded. Image shape: {image_shape}")
    # create the generator
    generator_config = config["generator_config"]
    generator = FCNGenerator(image_shape=image_shape, **generator_config)
    logger.info("Generator created.")
    # create the discriminator
    discriminator_config = config["discriminator_config"]
    discriminator = FCNDiscriminator(image_shape=image_shape, **discriminator_config)
    logger.info("Discriminator created.")
    # create wandb
    wandb_config = config["wandb_config"]
    if wandb_config:
        logger.info("Creating wandb logger.")
        wandb_config["config"] = config
        wandb_run = create_wandb_logger(**wandb_config)
    else:
        logger.info("Wandb logger not created.")
        wandb_run = None
    # create the GAN
    gan_config = config["gan_config"]
    gan_config_for_train_keys = [
        "epochs",
        "max_iteration_per_epoch",
        "log_interval",
        "image_plot_interval",
        "image_save_path",
        "image_save_interval",
        "model_save_path",
    ]
    gan_config_for_train = {
        k: v for k, v in gan_config.items() if k in gan_config_for_train_keys
    }
    gan_config_for_train["dataset"] = dataset
    gan_config_for_init = {
        k: v for k, v in gan_config.items() if k not in gan_config_for_train_keys
    }
    gan = FCNGAN(generator, discriminator, wandb_run=wandb_run, **gan_config_for_init)
    logger.info("FCNGAN created.")
    # train the GAN
    gan.train(**gan_config_for_train)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
