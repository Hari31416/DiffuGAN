from DiffuGAN.utils import (
    create_simple_logger,
    parse_activation,
    show_image,
    create_grid_from_batched_image,
)
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import wandb


class FNNGenerator(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        img_shape: tuple[int],
        layer_sizes: list[int] = [128, 256, 512],
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "sigmoid",
    ):
        """Initializes the FNNGenerator

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent space
        img_shape : tuple[int]
            The shape of the image that the generator will generate
        layer_sizes : list[int], optional
            The sizes of the layers in the generator, by default [128, 256, 512]
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        """
        super(FNNGenerator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation = parse_activation(final_activation)
        self.logger = create_simple_logger("FNNGenerator")
        self.model = self.build_module()

    def build_module(self):
        """Builds the generator module
        The generator is a simple feedforward neural network with number of units as specified in `layer_sizes`. We have used only linear layers and batchnorm layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        layer_sizes = (
            [self.latent_dim] + self.layer_sizes + [int(np.prod(self.img_shape))]
        )

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            self.logger.debug(
                f"Added Linear layer with in_features={in_features} and out_features={out_features}"
            )
            # add batchnorm after each layer if not the last or first layer
            if i < len(layer_sizes) - 2 and i > 0:
                layers.append(nn.BatchNorm1d(out_features))
                self.logger.debug(
                    f"Added BatchNorm1d layer with num_features={out_features}"
                )
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")

        # add final layer
        layers.append(self.final_activation)
        self.logger.debug(f"Added {self.final_activation} layer")

        return nn.Sequential(*layers)

    def forward(self, z):
        """Does a forward pass through the generator

        Parameters
        ----------
        z : torch.Tensor
            The input tensor to the generator. Must be of shape `(batch_size, latent_dim)`

        Returns
        -------
        torch.Tensor
            The output tensor of the generator. Must be of shape `(batch_size, *img_shape)`
        """
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def generate(self, z, show=False, return_tensor=False):
        img = self(z)
        numpy_image = img.detach().cpu().numpy()
        if show:
            grid = create_grid_from_batched_image(
                numpy_image,
                pad_value=1,
                return_type="numpy",
            )
            show_image(grid, title="Generated Images")
        if return_tensor:
            return img
        return numpy_image


class FNNDiscriminator(nn.Module):
    """A simple feedforward neural network discriminator for GANs"""

    def __init__(
        self,
        img_shape: tuple[int],
        layer_sizes: list[int] = [512, 256, 128],
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "sigmoid",
        dropout_rates: Union[list[float], float] = 0.3,
    ):
        """Initializes the FNNDiscriminator

        Parameters
        ----------
        img_shape : tuple[int]
            The shape of the image that the discriminator will take as input
        layer_sizes : list[int], optional
            The sizes of the layers in the discriminator, by default [512, 256, 128]
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        dropout_rates : Union[list[float], float], optional
            The dropout rates to use in the discriminator, by default 0.3. If a float is passed, the same dropout rate is used for all layers except the first and last layers.

            Note that the length of the list should be `len(layer_sizes) - 1` since the first layer does not have a dropout layer and the last layer has the final activation function.
        """
        super(FNNDiscriminator, self).__init__()

        self.img_shape = img_shape
        self.layer_sizes = layer_sizes
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation = parse_activation(final_activation)
        self.logger = create_simple_logger("FNNDiscriminator")
        if isinstance(dropout_rates, float):
            self.dropout_rates = [dropout_rates] * (len(layer_sizes) - 1)
        else:
            self.dropout_rates = dropout_rates
        self.model = self.build_module()

    def build_module(self):
        """Builds the discriminator module
        The discriminator is a simple feedforward neural network with number of units as specified in `layer_sizes`. We have used only linear layers and dropout layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        layer_sizes = [int(np.prod(self.img_shape))] + self.layer_sizes + [1]

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            self.logger.debug(
                f"Added Linear layer with in_features={in_features} and out_features={out_features}"
            )
            # add batchnorm after each layer if not the last or first layer
            if i < len(layer_sizes) - 2 and i > 0:
                layers.append(nn.Dropout(self.dropout_rates[i - 1]))
                self.logger.debug(
                    f"Added Dropout layer with p={self.dropout_rates[i-1]}"
                )
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")

        # add final layer
        layers.append(self.final_activation)
        self.logger.debug(f"Added {self.final_activation} layer")

        return nn.Sequential(*layers)

    def forward(self, img):
        """Does a forward pass through the discriminator

        Parameters
        ----------
        img : torch.Tensor
            The input tensor to the discriminator. Must be of shape `(batch_size, *img_shape)`

        Returns
        -------
        torch.Tensor
            The output tensor of the discriminator. Must be of shape `(batch_size, 1)`
        """
        img = img.view(img.size(0), -1)
        return self.model(img)


class GAN:
    """A class to implement a simple GAN model. The model is trained using the ADAM optimizer and the BCE loss function."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        k: int = 1,
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
    ):
        """Initializes the GAN model

        Parameters
        ----------
        generator : nn.Module
            The generator model
        discriminator : nn.Module
            The discriminator model
        k : int, optional
            The number of iterations to train the discriminator for every iteration of the generator, by default 1
        wandb_run : Union[wandb.sdk.wandb_run.Run, None], optional
            The wandb run object to log the training, by default None
        """
        self.logger = create_simple_logger("GAN")
        self.generator = generator
        self.discriminator = discriminator
        self.k = k
        self.bce_loss = nn.BCELoss()
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        if wandb_run is not None:
            self.use_wandb = True
            self.wandb_run = wandb_run
        else:
            self.use_wandb = False

    def generator_loss(self, fake: torch.Tensor):
        """Calculates the generator loss"""
        fake_labels = torch.ones(fake.size(0), 1)
        y_pred = self.discriminator(fake)
        return self.bce_loss(y_pred, fake_labels)

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor):
        """Calculates the discriminator loss"""
        real_labels = torch.ones(real.size(0), 1)
        fake_labels = torch.zeros(fake.size(0), 1)
        concat_images = torch.cat((real, fake), 0)
        concat_labels = torch.cat((real_labels, fake_labels), 0)
        y_pred = self.discriminator(concat_images)
        d_loss = self.bce_loss(y_pred, concat_labels)
        return d_loss

    def log_to_wandb(self, d_loss, g_loss, real_imgs, noises):
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
        real_imgs = real_imgs[:25]
        real_imgs = real_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
        real_wandb = self.wandb_run.Image(
            create_grid_from_batched_image(real_imgs, return_type="numpy")
        )
        fake_images = self.generator(noises)
        fake_images = fake_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        fake_wandb = self.wandb_run.Image(
            create_grid_from_batched_image(fake_images, return_type="numpy")
        )
        self.wandb_run.log(
            {
                "real images": real_wandb,
                "fake images": fake_wandb,
            }
        )

    def train(
        self,
        dataset: torch.utils.data.DataLoader,
        epochs: int,
        max_iteration_per_epoch: Union[int, None] = None,
        log_interval: int = 100,
    ):
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

        Returns
        -------
        None
        """
        losses = {"D": [], "G": []}
        noises = torch.randn(49, self.generator.latent_dim)
        # this is to have the same images for each epoch while logging
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataset):
                batch_size = imgs.size(0)
                real_imgs = imgs
                z = torch.randn(batch_size, self.generator.latent_dim)
                fake_imgs = self.generator(z)
                self.optimizer_D.zero_grad()
                d_loss = self.discriminator_loss(
                    real_imgs, fake_imgs.detach()
                )  # need to detach the fake_imgs tensor to avoid backpropagating through the generator
                d_loss.backward()
                self.optimizer_D.step()

                if i % self.k == 0:
                    self.optimizer_G.zero_grad()
                    g_loss = self.generator_loss(fake_imgs)
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % log_interval == 0:
                    d_loss = d_loss.item()
                    g_loss = g_loss.item()
                    losses["D"].append(d_loss)
                    losses["G"].append(g_loss)
                    self.logger.info(
                        f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataset)}] [D loss: {d_loss} | G loss: {g_loss}]"
                    )
                    if self.use_wandb:
                        self.log_to_wandb(d_loss, g_loss, real_imgs, noises)

                if max_iteration_per_epoch is not None and i >= max_iteration_per_epoch:
                    break

        self.logger.info("Training complete")
        if self.use_wandb:
            self.wandb_run.finish()
