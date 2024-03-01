from DiffuGAN.GAN.networks import BaseDiscriminator, BaseGenerator, BaseGAN
from typing import Union
import torch
import torch.nn as nn
import numpy as np
import wandb
import logging
import os

CUR_DIR = os.getcwd()


class FCNGenerator(BaseGenerator):
    """A simple feedforward neural network generator for GANs"""

    def __init__(
        self,
        latent_dimension: int,
        image_shape: tuple[int],
        layer_sizes: list[int] = [128, 256, 512],
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "tanh",
        logger: Union[None, logging.Logger] = None,
    ) -> None:
        """Initializes the FCNGenerator

        Parameters
        ----------
        latent_dimension : int
            The dimension of the latent space
        image_shape : tuple[int]
            The shape of the image that the generator will generate
        layer_sizes : list[int], optional
            The sizes of the layers in the generator, by default [128, 256, 512]
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        logger : Union[None, logging.Logger], optional
            The logger to use, by default None. If None, a simple logger is created.
        """
        super(FCNGenerator, self).__init__(
            latent_dimension,
            image_shape,
            activation,
            activation_kwargs,
            final_activation,
            logger,
        )
        self.layer_sizes = layer_sizes
        self.model = self.build_module()

    def build_module(self) -> nn.Sequential:
        """Builds the generator module
        The generator is a simple feedforward neural network with number of units as specified in `layer_sizes`. We have used only linear layers and batchnorm layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        layer_sizes = (
            [self.latent_dimension]
            + self.layer_sizes
            + [int(np.prod(self.image_shape))]
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
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img


class FCNDiscriminator(BaseDiscriminator):
    """A simple feedforward neural network discriminator for GANs"""

    def __init__(
        self,
        image_shape: tuple[int],
        layer_sizes: list[int] = [512, 256, 128],
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
        layer_sizes : list[int], optional
            The sizes of the layers in the discriminator, by default [512, 256, 128]
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        dropout_rate : float, optional
            The dropout rate to use in the discriminator, by default 0.3
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        logger : Union[None, logging.Logger], optional
            The logger to use, by default None. If None, a simple logger is created.
        """
        super(FCNDiscriminator, self).__init__(
            image_shape,
            activation,
            activation_kwargs,
            dropout_rate,
            final_activation,
            logger,
        )

        self.layer_sizes = layer_sizes
        self.model = self.build_module()

    def build_module(self) -> nn.Sequential:
        """Builds the discriminator module
        The discriminator is a simple feedforward neural network with number of units as specified in `layer_sizes`. We have used only linear layers and dropout layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        self.dropout_rates = [self.dropout_rate] * (len(self.layer_sizes) - 1)
        layer_sizes = [int(np.prod(self.image_shape))] + self.layer_sizes + [1]

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            self.logger.debug(
                f"Added Linear layer with in_features={in_features} and out_features={out_features}"
            )
            # add batchnorm and dropout after each layer if not the last or first layer
            if i < len(layer_sizes) - 2 and i > 0:
                layers.append(nn.BatchNorm1d(out_features))
                self.logger.debug(
                    f"Added BatchNorm1d layer with num_features={out_features}"
                )
                layers.append(nn.Dropout(self.dropout_rates[i - 1]))
                self.logger.debug(
                    f"Added Dropout layer with p={self.dropout_rates[i-1]}"
                )
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")

        # add final layer
        if self.final_activation is not None:
            layers.append(self.final_activation)
            self.logger.debug(f"Added {self.final_activation} layer")

        return nn.Sequential(*layers)

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
        img = img.view(img.size(0), -1)
        return self.model(img)


class FCNGAN(BaseGAN):
    """A class to implement a simple GAN model. The model is trained using the ADAM optimizer and the BCE loss function."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        k: int = 1,
        generator_optimizer: str = "adam",
        discriminator_optimizer: str = "adam",
        generator_optimizer_kwargs: dict = {"lr": 0.002},
        discriminator_optimizer_kwargs: dict = {"lr": 0.002},
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
        config: dict = {},
        generator_loss_to_use: str = "bce",
        max_step_for_og_loss: int = 100,
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
            The keyword arguments to pass to the generator optimizer, by default `{"lr": 0.002}`
        discriminator_optimizer_kwargs : dict, optional
            The keyword arguments to pass to the discriminator optimizer, by default `{"lr": 0.002}`
        generator_loss_to_use : str, optional
            The loss function to use for the generator, by default "bce"

            - "og": Original GAN loss, which is `L = log(1 - D(G(z)))`
            - "bce": Binary Cross Entropy loss which is `L = -log(D(G(z)))`
            - "og_bce": Original GAN loss for the first `max_step_for_og_loss` steps and then BCE loss after that

        max_step_for_og_loss : int, optional
            The maximum number of steps to use the original GAN loss, by default 100
        wandb_run : Union[wandb.sdk.wandb_run.Run, None], optional
            The wandb run object to log the training, by default None
        config : dict, optional
            The configuration dictionary, by default {}. This will be passed to the wandb run object if it is not None and will be saved locally if the model is saved
        """
        super(FCNGAN, self).__init__(
            generator,
            discriminator,
            k,
            generator_optimizer,
            discriminator_optimizer,
            generator_optimizer_kwargs,
            discriminator_optimizer_kwargs,
            wandb_run,
            config,
        )
        self.generator_loss_to_use = generator_loss_to_use
        self.max_step_for_og_loss = max_step_for_og_loss

    def _generator_og_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Calculates the loss function using:
        L = log(1 - D(G(z)))
        """
        y_pred = self.discriminator(fake)
        L = torch.log(1 - y_pred)
        loss = torch.mean(L)
        return loss

    def _generator_bce_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Calculates the generator loss"""
        fake_labels = torch.ones(fake.size(0), 1)
        y_pred = self.discriminator(fake)
        return self.bce_loss(y_pred, fake_labels)

    def generator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Calculates the generator loss kwargs has the following

        - generator_loss_to_use: str
            The loss function to use for the generator

            - "og": Original GAN loss, which is `L = log(1 - D(G(z)))`
            - "bce": Binary Cross Entropy loss which is `L = -log(D(G(z)))`
            - "og_bce": Original GAN loss for the first `max_step_for_og_loss` steps and then BCE loss after that
        - step_count: int
            The current step count. This is used to switch between the two loss functions
        - max_step_for_og_loss: int
            The maximum number of steps to use the original GAN loss
        """
        if self.generator_loss_to_use == "og":
            func = self._generator_og_loss
        elif self.generator_loss_to_use == "bce":
            func = self._generator_bce_loss
        elif self.generator_loss_to_use == "og_bce":
            if self.step_count < self.max_step_for_og_loss:
                func = self._generator_og_loss
            else:
                func = self._generator_bce_loss
        return func(fake)

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the discriminator loss"""
        real_labels = torch.ones(real.size(0), 1)
        fake_labels = torch.zeros(fake.size(0), 1)
        concat_images = torch.cat((real, fake), 0)
        concat_labels = torch.cat((real_labels, fake_labels), 0)
        y_pred = self.discriminator(concat_images)
        d_loss = self.bce_loss(y_pred, concat_labels)
        return d_loss
