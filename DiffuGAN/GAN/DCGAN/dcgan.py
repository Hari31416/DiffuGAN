from DiffuGAN.GAN.WGAN import WGAN
from DiffuGAN.GAN.GAN import FCNGAN
from DiffuGAN.GAN.networks import BaseGenerator, BaseDiscriminator

import torch.nn as nn
from typing import Union
from math import prod
import logging
import wandb


class ConvGenerator(BaseGenerator):
    """Creates a simple convolutional generator for GANs. The generator takes a latent vector as input and generates an image. The generator uses a series of convolutional layers followed by upscaling layers to generate the image. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""

    def __init__(
        self,
        latent_dimension: int,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        projection_filters: int = 1024,
        convolution_filters: Union[list[int], str] = "auto",
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "tanh",
        logger: Union[None, logging.Logger] = None,
    ) -> None:
        """Initializes the ConvGenerator

        Parameters
        ----------
        latent_dimension : int
            The dimension of the latent space
        image_shape : tuple[int, int, int], optional
            The shape of the image that the generator will generate, by default (1, 28, 28)
        projection_filters : int, optional
            The number of channels in the projection layer, by default 1024
        convolution_filters : list[int], or str, optional
            The number of filters for each convolutional layer, by default [32, 64]. Note that the maximum length of the list should be equal to the maximum number of times the image can be doubled since the image is doubled at each convolutional layer. If the string "auto" is passed, the number of filters for each convolutional layer is automatically calculated based on the image size and the projection channels. The number of filters is halved at each step.
        activation : str, optional
            The activation function to use, by default "relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {}
        final_activation : str, optional
            The final activation function to use, by default "tanh"
        logger : Union[None, logging.Logger], optional
            The logger to use, by default None
        """
        super(ConvGenerator, self).__init__(
            latent_dimension,
            image_shape,
            activation,
            activation_kwargs,
            final_activation,
            logger,
        )
        self.projection_filters = projection_filters
        if (
            isinstance(convolution_filters, str)
            and convolution_filters.lower() == "auto"
        ):
            self.convolution_filters = self._get_auto_channels(
                image_shape, projection_filters
            )
            self.logger.info(
                f"Using auto convolution channels: {self.convolution_filters}"
            )
        else:
            self.convolution_filters = convolution_filters
            self.validate_conv_layers_list(image_shape, convolution_filters)
        self.model = self.build_module()

    def _get_maximum_doubles(self, image_shape: tuple[int, int, int]) -> int:
        """Get the maximum number of times the image can be doubled"""
        _, h, w = image_shape
        min_dim = min(h, w)
        max_twos = 0
        while min_dim % 2 == 0:
            min_dim = min_dim // 2
            max_twos += 1
        # since the last convolution layer must increase the image size by a factor of 2,
        # the maximum number of conv layers is the highest power of 2 that is less than the image size
        max_twos -= 1
        self.logger.debug(f"Maximum number of possible conv layers is {max_twos}")
        return max_twos

    def _get_auto_channels(
        self, image_shape: tuple[int, int, int], projection_filters: int
    ) -> list[int]:
        max_twos = self._get_maximum_doubles(image_shape)
        # the auto channels are calculated by halving the projection channels at each step
        return [projection_filters // (2 ** (i + 1)) for i in range(max_twos)]

    def find_same_padding(self, kernel_size: int, stride: int) -> int:
        """Finds the padding required to keep the image size the same after convolution"""
        return (kernel_size - stride) // 2

    def validate_conv_layers_list(
        self, image_shape: tuple[int, int, int], convolution_filters: list[int]
    ) -> None:
        """Validates the conv layers list to ensure that the image size is compatible with the number of conv layers. If the image size is not compatible, the number of conv layers is adjusted accordingly.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image
        convolution_filters : list[int]
            The list of number of filters for each convolutional layer

        Raises
        ------
        ValueError
            If the image size is not compatible with the number of conv layers

        Notes
        -----
        The image size should be divisible by 2 otherwise the upsampling would not give the desired output. Make sure that the image size is divisible by 2 or use a different generator. The number of conv filters should be equal to the maximum number of times the image can be doubled minus 1 since the image is doubled at each convolutional layer. If the number of conv layers is not compatible with the image size, the number of conv layers is adjusted accordingly.
        """
        min_dim = min(image_shape[1], image_shape[2])
        max_twos = self._get_maximum_doubles(image_shape)
        # if max_power is 0, meaning that the image size is not divisible by 2, raise an error
        if max_twos < 1:
            msg = f"The image size is too small to be used with a convolutional generator or the image dimension is not divisible by 2. Please use a different image size."
            self.logger.error(msg)
            raise ValueError(msg)

        # Write a warning if the number of conv layers is not compatible with the image size
        if len(self.convolution_filters) > max_twos:
            self.logger.warning(
                f"The lowest dimension of the image is {min_dim}. The maximum number of times the image can be halved is {max_twos} and hence the number of conv layers should be {max_twos} but you have provided {convolution_filters}. The convolution_filters reduced to a maximum of {max_twos} to match the image size."
            )
            self.num_conv_layers = max_twos
            self.convolution_filters = convolution_filters[:max_twos]

    def build_module(self) -> nn.Sequential:
        """Builds the generator module. The module consists of a series of linear layers followed by a series of convolutional layers and upscaling layers. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""
        c, w, h = self.image_shape
        num_conv_block = len(self.convolution_filters)
        projection_h = w // (2 ** (num_conv_block + 1))
        projection_w = h // (2 ** (num_conv_block + 1))
        reshape_shape = (self.projection_filters, projection_h, projection_w)
        self.logger.debug(f"Reshape shape: {reshape_shape}")

        linear_layer_shape = prod(reshape_shape)
        linear_layer = nn.Linear(self.latent_dimension, linear_layer_shape)
        reshape_layer = nn.Unflatten(1, reshape_shape)
        temp_channels = [self.projection_filters] + self.convolution_filters
        layers = [
            linear_layer,
            self.activation,
            reshape_layer,
        ]
        self.logger.debug(
            f"Added Linear layer with in_features={self.latent_dimension} and out_features={linear_layer_shape}"
        )
        self.logger.debug(f"Added {self.activation} layer")
        self.logger.debug(f"Added Unflatten layer with shape {reshape_shape}")

        for i in range(num_conv_block):
            # this conv layer preserves the image size
            conv_layer = nn.ConvTranspose2d(
                temp_channels[i],
                temp_channels[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
            )
            batch_norm = nn.BatchNorm2d(temp_channels[i + 1])
            layers.append(conv_layer)
            self.logger.debug(
                f"Added Conv2d layer with in_channels={temp_channels[i]} and out_channels={temp_channels[i+1]}"
            )
            layers.append(batch_norm)
            self.logger.debug(
                f"Added BatchNorm2d layer with num_features={temp_channels[i+1]}"
            )
            layers.append(self.activation)
            self.logger.debug(f"Added {self.activation} layer")
        # add a projection layer to get the image shape
        layers.append(
            nn.ConvTranspose2d(
                temp_channels[-1],
                c,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.logger.debug(
            f"Added projection layer with in_channels={temp_channels[-1]} and out_channels={self.image_shape[0]}"
        )
        # finally add the activation layer
        layers.append(self.final_activation)
        self.logger.debug(f"Added {self.final_activation} layer")
        return nn.Sequential(*layers)

    def forward(self, x):
        """Does a forward pass through the generator

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the generator. Must be of shape `(batch_size, latent_dimension)`

        Returns
        -------
        torch.Tensor
            The output tensor of the generator. Must be of shape `(batch_size, *img_shape)`
        """
        return self.model(x)


class ConvDiscriminator(BaseDiscriminator):
    """Creates a simple convolutional discriminator for GANs. The discriminator takes an image as input and outputs a single value. The discriminator uses a series of convolutional layers followed by downscaling layers to generate the output. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        convolution_filters: list[int] = [32, 64],
        kernel_size: int = 3,
        activation: str = "relu",
        activation_kwargs: dict = {},
        dropout_rate: float = 0.3,
        downsample_strategy: str = "same",
        logger: Union[None, logging.Logger] = None,
        final_activation: str = "sigmoid",
    ):
        """Initializes the ConvDiscriminator

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image that the discriminator will take as input
        convolution_filters : list[int], optional
            The sizes of the layers in the discriminator, by default [32, 64]
        activation : str, optional
            The activation function to use, by default "leaky_relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {"negative_slope": 0.2}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        dropout_rate : Union[list[float], float], optional
            The dropout rates to use in the discriminator, by default 0.3.
        downsample_strategy : str, optional
            The strategy to use for downsampling the image, by default "same".
            The options are:

            - "same": Use a convolutional layer with stride 2 to downsample the image
            - "separate": Use a convolutional layer with stride 1 followed by another convolutional layer with stride 2 to downsample the image
        """
        super(ConvDiscriminator, self).__init__(
            image_shape,
            activation,
            activation_kwargs,
            dropout_rate,
            final_activation,
            logger,
        )
        self.convolution_filters = convolution_filters
        self.num_conv_layers = len(self.convolution_filters)
        self.validate_conv_layers_list(image_shape, convolution_filters)
        if kernel_size % 2 == 0:
            msg = "Kernel size must be odd."
            self.logger.error(msg)
            raise ValueError(msg)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.downscale_method = downsample_strategy
        self.model = self.build_module()

    def validate_conv_layers_list(
        self, image_shape: tuple[int, int, int], convolution_filters: list[int]
    ):
        """Validates the conv layers list to ensure that the image size is compatible with the number of conv layers. If the image size is not compatible, the number of conv layers is adjusted accordingly.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image
        convolution_filters : list[int]
            The list of number of filters for each convolutional layer

        Raises
        ------
        ValueError
            If the image size is not compatible with the number of conv layers

        Notes
        -----
        The image size should be divisible by 2 otherwise the upsampling would not give the desired output. Make sure that the image size is divisible by 2 or use a different generator. The number of conv filters should be equal to the maximum number of times the image can be doubled since the image is doubled at each convolutional layer. If the number of conv layers is not compatible with the image size, the number of conv layers is adjusted accordingly.
        """
        _, h, w = image_shape
        min_dim = min(h, w)
        max_twos = 0
        while min_dim % 2 == 0:
            min_dim = min_dim // 2
            max_twos += 1

        self.logger.debug(f"Maximum number of possible conv layers is {max_twos}")
        # Write a warning if the number of conv layers is not compatible with the image size
        if self.num_conv_layers > max_twos:
            self.logger.warning(
                f"The maximum number of times the image can be halved is {max_twos} and hence the number of conv layers should be {max_twos} but you have provided {convolution_filters}. The convolution_filters will be reduced to a maximum of {max_twos} to avoid RuntimeError."
            )
            self.num_conv_layers = max_twos
            self.convolution_filters = convolution_filters[:max_twos]

    def build_module(self):
        """Builds the discriminator module
        The discriminator is a simple feedforward neural network with number of units as specified in `convolution_filters`. We have used only linear layers and dropout layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        c, w, h = self.image_shape
        final_conv_kernel = w // (2**self.num_conv_layers)
        convolution_filters = [c] + self.convolution_filters

        for i in range(len(convolution_filters) - 1):
            # this conv layer preserves the image size
            conv_layer_same_shape = nn.Conv2d(
                in_channels=convolution_filters[i],
                out_channels=convolution_filters[i + 1],
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
            )
            conv_layer_downsample = nn.Conv2d(
                in_channels=convolution_filters[i],
                out_channels=convolution_filters[i + 1],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            )
            conv_layer_downsample_only = nn.Conv2d(
                in_channels=convolution_filters[i + 1],
                out_channels=convolution_filters[i + 1],
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            )
            dropout = nn.Dropout2d(self.dropout_rate)

            batch_norm = nn.BatchNorm2d(convolution_filters[i + 1])

            if self.downscale_method == "same":
                layers.append(conv_layer_downsample)
                self.logger.debug(
                    f"Added Conv2d layer with in_channels={convolution_filters[i]} and out_channels={convolution_filters[i+1]} and stride=2"
                )
                layers.append(batch_norm)
                self.logger.debug(
                    f"Added BatchNorm2d layer with num_features={convolution_filters[i+1]}"
                )
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")
                # add dropout layer
                layers.append(dropout)
                self.logger.debug(f"Added Dropout2d layer with p={self.dropout_rate}")
            elif self.downscale_method == "separate":
                layers.append(conv_layer_same_shape)
                self.logger.debug(
                    f"Added Conv2d layer with in_channels={convolution_filters[i]} and out_channels={convolution_filters[i+1]} and stride=1"
                )
                layers.append(batch_norm)
                self.logger.debug(
                    f"Added BatchNorm2d layer with num_features={convolution_filters[i+1]}"
                )
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")
                layers.append(dropout)
                self.logger.debug(f"Added Dropout2d layer with p={self.dropout_rate}")
                layers.append(conv_layer_downsample_only)
                self.logger.debug(
                    f"Added Conv2d layer with in_channels={convolution_filters[i+1]} and out_channels={convolution_filters[i+1]} and stride=2"
                )
                layers.append(batch_norm)
                self.logger.debug(
                    f"Added BatchNorm2d layer with num_features={convolution_filters[i+1]}"
                )
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")
            else:
                msg = f"Downscale method {self.downscale_method} is not supported. Please use 'same' or 'separate'."
                self.logger.error(msg)
                raise ValueError(msg)

        # add the final conv layer to make the output a single value
        final_conv = nn.Conv2d(
            in_channels=convolution_filters[-1],
            out_channels=1,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0,
        )
        layers.append(final_conv)
        self.logger.debug(
            f"Added final Conv2d layer with in_channels={convolution_filters[-1]} and out_channels=1"
        )
        layers.append(self.final_activation)
        self.logger.debug(f"Added {self.final_activation} layer")
        return nn.Sequential(*layers)

    def forward(self, x):
        """Does a forward pass through the discriminator

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the discriminator. Must be of shape `(batch_size, *image_shape)`

        Returns
        -------
        torch.Tensor
            The output tensor of the discriminator. Must be of shape `(batch_size, 1)`
        """
        y = self.model(x)
        # add the batch dimension
        y = y.view(-1, 1)
        return y


class DCGAN(FCNGAN):

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
    ):
        super().__init__(
            generator,
            discriminator,
            k,
            generator_optimizer,
            discriminator_optimizer,
            generator_optimizer_kwargs,
            discriminator_optimizer_kwargs,
            wandb_run,
            config,
            generator_loss_to_use,
            max_step_for_og_loss,
        )


class WDCGAN(WGAN):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        k: int = 5,
        generator_optimizer: str = "rmsprop",
        discriminator_optimizer: str = "rmsprop",
        generator_optimizer_kwargs: dict = {},
        discriminator_optimizer_kwargs: dict = {},
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
        config: dict = {},
        c: float = 0.01,
        **kwargs,
    ):
        super().__init__(
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
        self.c = c
