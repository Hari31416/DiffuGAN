from DiffuGAN.utils import (
    create_simple_logger,
    parse_activation,
    parse_optimizer,
    create_grid_from_batched_image,
    ImagePlotter,
)
import torch
import torch.nn as nn
import numpy as np
from typing import Union
import logging
import wandb
import os
import yaml

CUR_DIR = os.getcwd()


class BaseGenerator(nn.Module):
    """A base class for all the generators in the GANs"""

    def __init__(
        self,
        latent_dimension: int,
        image_shape: tuple[int],
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "tanh",
        logger: Union[None, logging.Logger] = None,
    ) -> None:
        """Initializes the Generator

        Parameters
        ----------
        latent_dimension : int
            The dimension of the latent space
        image_shape : tuple[int]
            The shape of the image that the generator will generate
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        logger : Union[None, logging.Logger], optional
            The logger to use, by default None. If None, a simple logger is created.
        """
        super(BaseGenerator, self).__init__()

        self.logger = logger or create_simple_logger(self.__class__.__name__)
        self.image_shape = image_shape
        self.latent_dimension = latent_dimension
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation = parse_activation(final_activation)

    def build_module(self):
        """Builds the generator module"""
        msg = "The build_module method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Does a forward pass through the generator

        Parameters
        ----------
        z : torch.Tensor
            The input tensor to the generator. Must be of shape `(batch_size, latent_dimension)`

        Returns
        -------
        torch.Tensor
            The output tensor of the generator. Must be of shape `(batch_size, *image_shape)`
        """
        msg = "The forward method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def generate(self, z: torch.Tensor, return_tensor: bool = False):
        """Generates the images from the noise vectors. Makes sure that the output is between 0 and 1."""
        self.model.eval()
        img = self(z)
        if return_tensor:
            return img
        # permute the dimensions to make it suitable for plotting
        img = img.permute(0, 2, 3, 1)
        numpy_image = img.detach().cpu().numpy()
        # normalize the image
        numpy_image = (numpy_image - numpy_image.min()) / (
            numpy_image.max() - numpy_image.min()
        )

        return numpy_image


class BaseDiscriminator(nn.Module):
    """A base class for all the discriminators in the GANs"""

    def __init__(
        self,
        image_shape: tuple[int],
        activation: str = "LeakyReLU",
        activation_kwargs: dict = {"negative_slope": 0.2},
        dropout_rate: float = 0.3,
        final_activation: str = "sigmoid",
        logger: Union[None, logging.Logger] = None,
    ) -> None:
        """Initializes the FCNDiscriminator

        Parameters
        ----------
        image_shape : tuple[int]
            The shape of the image that the discriminator will take as input
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        dropout_rates : float, optional
            The dropout rate to use in the discriminator, by default 0.3
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        logger : Union[None, logging.Logger], optional
            The logger to use, by default None. If None, a simple logger is created.
        """
        super(BaseDiscriminator, self).__init__()

        self.logger = logger or create_simple_logger(self.__class__.__name__)
        self.image_shape = image_shape
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation = parse_activation(final_activation)
        self.dropout_rate = dropout_rate

    def build_module(self) -> nn.Sequential:
        """Builds the discriminator module"""
        msg = "The build_module method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Does a forward pass through the discriminator

        Parameters
        ----------
        img : torch.Tensor
            The input tensor to the discriminator. Must be of shape `(batch_size, *image_shape)`

        Returns
        -------
        torch.Tensor
            The output tensor of the discriminator. Must be of shape `(batch_size, 1)`
        """
        msg = "The forward method must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)


class BaseGAN:
    """An abstract GAN class to be used as a base class for other GAN models"""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        k: int = 1,
        generator_optimizer: str = "adam",
        discriminator_optimizer: str = "adam",
        generator_optimizer_kwargs: dict = {"lr": 0.0001},
        discriminator_optimizer_kwargs: dict = {"lr": 0.0001},
        device: Union[str, None] = None,
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
        config: dict = {},
    ) -> None:
        """Initializes the GAN model

        Parameters
        ----------
        generator : nn.Module
            The generator model
        discriminator : nn.Module
            The discriminator model
        k : int, optional
            The number of iterations to train the discriminator for every iteration of the generator, by default 1
        generator_optimizer : str, optional
            The optimizer to use for the generator, by default "adam"
        discriminator_optimizer : str, optional
            The optimizer to use for the discriminator, by default "adam"
        generator_optimizer_kwargs : dict, optional
            The keyword arguments to pass to the generator optimizer, by default `{"lr": 0.0001}`
        discriminator_optimizer_kwargs : dict, optional
            The keyword arguments to pass to the discriminator optimizer, by default `{"lr": 0.0001}`
        device : Union[str, None], optional
            The device to use for training, by default None. If None, the device is automatically selected
        wandb_run : Union[wandb.sdk.wandb_run.Run, None], optional
            The wandb run object to log the training, by default None
        config : dict, optional
            The configuration dictionary, by default {}. This will be passed to the wandb run object if it is not None and will be saved locally if the model is saved
        """
        self.logger = create_simple_logger("GAN")
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger.info(f"Using device: {self.device}")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.k = k
        self.bce_loss = nn.BCELoss()
        self.optimizer_G = parse_optimizer(
            generator_optimizer, generator.parameters(), **generator_optimizer_kwargs
        )
        self.optimizer_D = parse_optimizer(
            discriminator_optimizer,
            discriminator.parameters(),
            **discriminator_optimizer_kwargs,
        )
        if wandb_run is not None:
            self.use_wandb = True
            self.wandb_run = wandb_run
            self.wandb_run.config.update(config)
        else:
            self.use_wandb = False
        self.step_count = 0  # global step count
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using the device: {self.device}")
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def generator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Calculates the generator loss. This is a dummy function and should be overridden in the child classes"""
        msg = "The generator_loss method should be implemented"
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the discriminator loss. Must be overridden in the child classes"""
        msg = "The discriminator_loss method should be implemented"
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def log_to_wandb(
        self,
        d_loss: torch.Tensor,
        g_loss: torch.Tensor,
        real_imgs: torch.Tensor,
        noises: torch.Tensor,
    ) -> None:
        """Logs the losses and some images to wandb

        Parameters
        ----------
        d_loss : torch.Tensor
            The discriminator loss
        g_loss : torch.Tensor
            The generator loss
        real_imgs : torch.Tensor
            The real images
        noises : torch.Tensor
            The noise vectors

        Returns
        -------
        None
        """
        self.wandb_run.log({"D loss": d_loss, "G loss": g_loss})
        # save some images
        max_images = min(25, real_imgs.size(0))
        real_imgs = real_imgs[:max_images]
        real_imgs = real_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
        real_wandb = self.wandb_run.Image(
            create_grid_from_batched_image(real_imgs, return_type="numpy")
        )
        fake_images = self.generator.generate(noises)
        fake_wandb = self.wandb_run.Image(
            create_grid_from_batched_image(fake_images, return_type="numpy")
        )
        self.wandb_run.log(
            {
                "real images": real_wandb,
                "fake images": fake_wandb,
            }
        )

    def _get_absolute_path(self, path: str) -> str:
        """Returns the absolute path of the specified path"""
        if path.startswith(os.path.sep):
            self.logger.debug(f"Using the absolute path: {path}")
            return path
        path = os.path.join(CUR_DIR, path)
        self.logger.debug(f"Using the relative path: {path}")
        return path

    def plot_generated_images(
        self,
        noises: torch.Tensor,
        plotter: ImagePlotter,
        epoch: int,
        batch_idx: int,
        image_save_path: Union[str, None],
        save_image: bool = False,
    ) -> None:
        """Plots the generated images

        Parameters
        ----------
        noises : torch.Tensor
            The noise vectors to generate the images
        plotter : ImagePlotter
            The ImagePlotter object
        epoch : int
            The current epoch
        batch_idx : int
            The current batch index
        image_save_path : Union[str, None]
            The path to save the images, by default None
        save_image : bool
            Whether to save the image or not

        Returns
        -------
        None
        """
        fake_images = self.generator.generate(noises)
        grid = create_grid_from_batched_image(
            fake_images,
            return_type="numpy",
        )
        title = f"Generated Images (Epoch: {epoch}, Batch: {batch_idx}, Step: {self.step_count})"
        if image_save_path and save_image:
            step_number = str(self.step_count).zfill(5)
            image_name = f"generated_images_{step_number}.png"
            # if the path is absolute, use it as is, else relative to the current file
            directory = self._get_absolute_path(image_save_path)
            os.makedirs(directory, exist_ok=True)
            path_to_save = os.path.join(directory, image_name)
        else:
            path_to_save = None

        plotter.update_image(grid, title=title, path_to_save=path_to_save)

    def save_models(self, path: str) -> None:
        """Saves the model to the specified path

        Parameters
        ----------
        path : str
            The path to save the model

        Returns
        -------
        None
        """
        path_to_save = self._get_absolute_path(path)
        self.logger.info(f"Saving the model to {path_to_save}")
        os.makedirs(path_to_save, exist_ok=True)
        model_file_name = os.path.join(path_to_save, "gan_models.pt")
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
            },
            model_file_name,
        )
        # save the config as yaml
        config_name = "config.yaml"
        config_path = os.path.join(path_to_save, config_name)
        config_to_yaml = yaml.dump(self.config)
        with open(config_path, "w") as f:
            f.write(config_to_yaml)

    def load_models(self, path: str) -> None:
        """Loads the model from the specified path

        Parameters
        ----------
        path : str
            The path to load the model from

        Returns
        -------
        None
        """
        path_to_load = self._get_absolute_path(path)
        file_to_load = os.path.join(path_to_load, "gan_models.pt")
        self.logger.info(f"Loading the model from {file_to_load}")
        checkpoint = torch.load(file_to_load)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        # add models to the device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def after_discriminator_update(self) -> None:
        """A function to be called after updating the discriminator. This can be overridden in the child classes"""
        pass

    def after_generator_update(self) -> None:
        """A function to be called after updating the generator. This can be overridden in the child classes"""
        pass

    def train(
        self,
        dataset: torch.utils.data.DataLoader,
        epochs: int,
        max_iteration_per_epoch: Union[int, None] = None,
        log_interval: int = 100,
        image_plot_interval: int = 0,
        image_save_path: Union[str, None] = None,
        image_save_interval: int = 1000,
        model_save_path: Union[str, None] = None,
    ) -> dict:
        """Trains the GAN model

        Parameters
        ----------
        dataset : torch.utils.data.DataLoader
            The dataset to train on
        epochs : int
            The number of epochs to train for
        max_iteration_per_epoch : Union[int, None], optional
            The maximum number of iterations to train for in each epoch, by default None
        log_interval : int, optional
            The interval at which to log the losses and images, by default 100
        image_plot_interval : int, optional
            The interval at which to plot the generated images. Images will not be plotted if this is 0, by default 0
        image_save_path : Union[str, None], optional
            The path to save the images, by default None. If None, the images will not be saved
        image_save_interval : int, optional
            The interval at which to save the images, by default 1000
        model_save_path : Union[str, None], optional
            The path to save the model, by default None. If this is not None, the model will be saved after training
        Returns
        -------
        None
        """
        losses = {"D": [], "G": []}
        # use the same noise vectors for each epoch for bettr comparison
        noises = torch.randn(49, self.generator.latent_dimension, device=self.device)
        plotter = ImagePlotter(figsize=(14, 14))
        if not image_plot_interval:
            self.logger.info("image_plot_interval is 0. Images will not be plotted")
            image_save_to_plot_ratio = 0
        else:
            image_save_to_plot_ratio = image_save_interval // image_plot_interval
        image_plot_count = 0
        # this is to have the same images for each epoch while logging
        for epoch in range(epochs):
            for batch_idx, (imgs, _) in enumerate(dataset):
                batch_size = imgs.size(0)
                real_imgs = imgs.to(self.device)
                z = torch.randn(
                    batch_size, self.generator.latent_dimension, device=self.device
                )
                fake_imgs = self.generator(z)
                self.optimizer_D.zero_grad()
                d_loss = self.discriminator_loss(
                    real_imgs, fake_imgs.detach()
                )  # need to detach the fake_imgs tensor to avoid backpropagating through the generator
                d_loss.backward()
                self.optimizer_D.step()
                self.after_discriminator_update()

                if batch_idx % self.k == 0:
                    self.optimizer_G.zero_grad()
                    g_loss = self.generator_loss(real=real_imgs, fake=fake_imgs)
                    g_loss.backward()
                    self.optimizer_G.step()
                    self.after_generator_update()

                if batch_idx % log_interval == 0:
                    d_loss = d_loss.item()
                    g_loss = g_loss.item()
                    losses["D"].append(d_loss)
                    losses["G"].append(g_loss)
                    msg = f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(dataset)}] [D loss: {d_loss} | G loss: {g_loss}]"
                    print(msg)
                    if self.use_wandb:
                        self.log_to_wandb(d_loss, g_loss, real_imgs, noises)
                if image_plot_interval and batch_idx % image_plot_interval == 0:
                    if (
                        image_save_to_plot_ratio
                        and image_plot_count % image_save_to_plot_ratio == 0
                    ):
                        save_image = True
                    else:
                        save_image = False
                    self.plot_generated_images(
                        noises,
                        plotter,
                        epoch,
                        batch_idx,
                        image_save_path,
                        save_image=save_image,
                    )
                    image_plot_count += 1

                if (
                    max_iteration_per_epoch is not None
                    and batch_idx >= max_iteration_per_epoch
                ):
                    break
                self.step_count += 1

        self.logger.info("Training complete")
        if self.use_wandb:
            self.wandb_run.finish()
        if model_save_path:
            self.save_models(model_save_path)
        return losses
