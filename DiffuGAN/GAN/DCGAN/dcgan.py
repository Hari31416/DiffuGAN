from DiffuGAN.utils import (
    create_simple_logger,
    parse_activation,
    create_grid_from_batched_image,
)
from DiffuGAN.GAN.GAN import GAN
from typing import Union
from math import prod
import torch.nn as nn

import numpy as np
import wandb


class ConvGenerator(nn.Module):
    """Creates a simple convolutional generator for GANs. The generator takes a latent vector as input and generates an image. The generator uses a series of convolutional layers followed by upscaling layers to generate the image. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""

    def __init__(
        self,
        latent_dimension: int,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        conv_filters: list[int] = [32, 64],
        kernel_size: int = 3,
        upscale_method: str = "transpose",
        upscale_mode: str = "bilinear",
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "sigmoid",
    ) -> None:
        """Initializes the ConvGenerator

        Parameters
        ----------
        latent_dimension : int
            The dimension of the latent space
        image_shape : tuple[int, int, int], optional
            The shape of the image that the generator will generate, by default (1, 28, 28)
        conv_filters : list[int], optional
            The number of filters for each convolutional layer, by default [32, 64]. Note that the maximum length of the list should be equal to the maximum number of times the image can be doubled since the image is doubled at each convolutional layer.
        upscale_method : str, optional
            The method to use for upscaling, by default "transpose"
        upscale_mode : str, optional
            The mode to use for upscaling, by default "bilinear". Only used if `upscale_method` is "upscale"
        """
        super().__init__()
        self.logger = create_simple_logger("ConvGenerator")
        self.latent_dimension = latent_dimension
        self.image_shape = image_shape
        self.conv_filters = conv_filters
        if kernel_size % 2 == 0:
            msg = "Kernel size must be odd."
            self.logger.error(msg)
            raise ValueError(msg)

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_conv_layers = len(self.conv_filters)
        self.validate_conv_layers_list(image_shape, conv_filters)
        self.upscale_method = upscale_method
        self.upscale_mode = upscale_mode
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation_str = final_activation
        self.layers = self.build_module()
        self.model = nn.Sequential(*self.layers)

    def find_same_padding(self, kernel_size: int, stride: int):
        """Finds the padding required to keep the image size the same after convolution"""
        return (kernel_size - stride) // 2

    def validate_conv_layers_list(
        self, image_shape: tuple[int, int, int], conv_filters: list[int]
    ):
        """Validates the conv layers list to ensure that the image size is compatible with the number of conv layers. If the image size is not compatible, the number of conv layers is adjusted accordingly.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image
        conv_filters : list[int]
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
        # get the highest power of 2 that is less than the image size
        min_dim = min(h, w)
        max_twos = 0
        while min_dim % 2 == 0:
            min_dim = min_dim // 2
            max_twos += 1
        self.logger.debug(f"Maximum number of possible conv layers is {max_twos}")
        # if max_power is 0, meaning that the image size is not divisible by 2, raise an error
        if max_twos < 1:
            msg = f"The image size is too small to be used with a convolutional generator or the image dimension is not divisible by 2. Please use a different image size or use a different generator."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.num_conv_layers > max_twos and 2**max_twos == min(h, w):
            msg = f"Using the current number of conv layers will result in the first conv layer taking an input of size 1. This is not recommended. Consider using a smaller number of conv layers."
            self.logger.warning(msg)

        # Write a warning if the number of conv layers is not compatible with the image size
        if self.num_conv_layers > max_twos:
            self.logger.warning(
                f"The lowest dimension of the image is {min_dim}. The maximum number of times the image can be halved is {max_twos} and hence the number of conv layers should be {max_twos} but you have provided {conv_filters}. The conv_filters reduced to a maximum of {max_twos} to match the image size."
            )
            self.num_conv_layers = max_twos
            self.conv_filters = conv_filters[:max_twos]

    def _get_upsample_layer(self, layer_type="upscale", mode=None, channels=None):
        """Get the upscale layer based on the type of upscaling required"""
        if layer_type == "upscale":
            if mode is None:
                msg = "Mode must be provided for upscaling."
                self.logger.error(msg)
                raise ValueError(msg)
            self.logger.debug(f"Using upscaling method {layer_type} with mode {mode}")
            return nn.Upsample(scale_factor=2, mode=mode)

        elif layer_type == "transpose":
            if channels is None:
                msg = "Channels must be provided for transpose convolution."
                self.logger.error(msg)
                raise ValueError(msg)
            self.logger.debug(
                f"Using upscaling method {layer_type} with channels {channels}"
            )
            # the kernel size and stride are set to 4 and 2 respectively to double the image size with a padding of 1
            # The channel size is preserved
            return nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        else:
            msg = f"Invalid layer type {layer_type}. Only upscale and transpose are supported."
            self.logger.error(msg)
            raise ValueError(msg)

    def build_module(self):
        """Builds the generator module. The module consists of a series of linear layers followed by a series of convolutional layers and upscaling layers. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""
        reshape_shape = (
            self.image_shape[0],
            self.image_shape[1] // 2**self.num_conv_layers,
            self.image_shape[2] // 2**self.num_conv_layers,
        )
        self.logger.debug(f"Reshape shape: {reshape_shape}")
        linear_layer_shape = prod(reshape_shape)
        linear_layer = nn.Linear(self.latent_dimension, linear_layer_shape)
        reshape_layer = nn.Unflatten(1, reshape_shape)
        conv_filters = [self.image_shape[0]] + self.conv_filters
        layers = [
            linear_layer,
            self.activation,
            reshape_layer,
        ]
        self.logger.debug(
            f"Added Linear layer with in_features={self.latent_dimension} and out_features={linear_layer_shape}"
        )
        self.logger.debug(f"Added Unflatten layer with shape {reshape_shape}")

        for i in range(self.num_conv_layers):
            # this conv layer preserves the image size
            conv_layer = nn.Conv2d(
                in_channels=conv_filters[i],
                out_channels=conv_filters[i + 1],
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
            )

            # this layer upscales the image
            up_scale_layer = self._get_upsample_layer(
                layer_type=self.upscale_method,
                mode=self.upscale_mode,
                channels=conv_filters[i + 1],
            )
            # Add batch norm layer
            batch_norm = nn.BatchNorm2d(conv_filters[i + 1])
            layers.append(conv_layer)
            self.logger.debug(
                f"Added Conv2d layer with in_channels={conv_filters[i]} and out_channels={conv_filters[i+1]}"
            )
            layers.append(self.activation)
            self.logger.debug(f"Added {self.activation} layer")
            layers.append(batch_norm)
            self.logger.debug(
                f"Added BatchNorm2d layer with num_features={conv_filters[i+1]}"
            )
            layers.append(up_scale_layer)
            self.logger.debug(f"Added upscaling layer")

            # add activation only for transpose layers
            if self.upscale_method == "transpose":
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")
        # add a projection layer to get the image shape
        layers.append(
            nn.Conv2d(
                in_channels=conv_filters[-1],
                out_channels=self.image_shape[0],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.logger.debug(
            f"Added projection layer with in_channels={conv_filters[-1]} and out_channels={self.image_shape[0]}"
        )
        # finally add the sigmoid activation
        layers.append(nn.Sigmoid())
        self.logger.debug(f"Added Sigmoid layer")
        return layers

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

    def generate(self, z, show=False, return_tensor=False):
        img = self(z)
        numpy_image = img.detach().cpu().numpy()
        if show:
            _ = create_grid_from_batched_image(
                numpy_image,
                pad_value=1,
                return_type="numpy",
                quick_plot=True,
            )
        if return_tensor:
            return img
        return numpy_image


class ConvDiscriminator(nn.Module):
    """Creates a simple convolutional discriminator for GANs. The discriminator takes an image as input and outputs a single value. The discriminator uses a series of convolutional layers followed by downscaling layers to generate the output. The final layer is a sigmoid layer to ensure that the output is between 0 and 1."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        conv_filters: list[int] = [32, 64],
        kernel_size: int = 3,
        downscale_method: str = "maxpool",
        activation: str = "relu",
        activation_kwargs: dict = {},
        final_activation: str = "sigmoid",
        dropout_rates: Union[list[float], float] = 0.3,
    ):
        """Initializes the ConvDiscriminator

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image that the discriminator will take as input
        conv_filters : list[int], optional
            The sizes of the layers in the discriminator, by default [32, 64]
        activation : str, optional
            The activation function to use, by default "leaky_relu"
        activation_kwargs : dict, optional
            The keyword arguments to pass to the activation function, by default {"negative_slope": 0.2}
        final_activation : str, optional
            The final activation function to use, by default "sigmoid"
        dropout_rates : Union[list[float], float], optional
            The dropout rates to use in the discriminator, by default 0.3. If a float is passed, the same dropout rate is used for all layers except the first and last layers.

            Note that the length of the list should be `len(conv_filters) - 1` since the first layer does not have a dropout layer and the last layer has the final activation function.
        """
        super(ConvDiscriminator, self).__init__()
        self.logger = create_simple_logger("ConvDiscriminator")
        self.image_shape = image_shape
        self.conv_filters = conv_filters
        self.num_conv_layers = len(self.conv_filters)
        self.validate_conv_layers_list(image_shape, conv_filters)
        self.activation = parse_activation(activation, **activation_kwargs)
        self.final_activation = parse_activation(final_activation)
        if kernel_size % 2 == 0:
            msg = "Kernel size must be odd."
            self.logger.error(msg)
            raise ValueError(msg)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.downscale_method = downscale_method
        self.dropout_rates = dropout_rates

        self.model = self.build_module()

    def validate_conv_layers_list(
        self, image_shape: tuple[int, int, int], conv_filters: list[int]
    ):
        """Validates the conv layers list to ensure that the image size is compatible with the number of conv layers. If the image size is not compatible, the number of conv layers is adjusted accordingly.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of the image
        conv_filters : list[int]
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
        # get the highest power of 2 that is less than the image size
        min_dim = min(h, w)
        max_twos = int(np.log2(min_dim))
        self.logger.debug(f"Maximum number of possible conv layers is {max_twos}")

        if self.num_conv_layers == max_twos:
            msg = f"Using the current number of conv layers will result in the last conv layer reducing the image size to 1. This is not recommended. Consider using a smaller number of conv layers."
            self.logger.warning(msg)

        # Write a warning if the number of conv layers is not compatible with the image size
        if self.num_conv_layers > max_twos:
            self.logger.warning(
                f"The maximum number of times the image can be halved is {max_twos} and hence the number of conv layers should be {max_twos} but you have provided {conv_filters}. The conv_filters will be reduced to a maximum of {max_twos} to avoid RuntimeError."
            )
            self.num_conv_layers = max_twos
            self.conv_filters = conv_filters[:max_twos]

    def _get_downscale_layer(self, layer_type="maxpool", channels=None):
        """Get the downscale layer based on the type of downscaling required"""
        if layer_type == "maxpool":
            self.logger.debug(f"Using downscaling method {layer_type}")
            return nn.MaxPool2d(kernel_size=2, stride=2)

        elif layer_type == "conv":
            if channels is None:
                msg = "Channels must be provided for convolutional downscaling."
                self.logger.error(msg)
                raise ValueError(msg)
            self.logger.debug(
                f"Using downscaling method {layer_type} with channels {channels}"
            )
            # the kernel size and stride are set to 3 and 2 respectively to halve the image size
            # The channel size is preserved
            return nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )

        else:
            msg = (
                f"Invalid layer type {layer_type}. Only maxpool and conv are supported."
            )
            self.logger.error(msg)
            raise ValueError(msg)

    def build_module(self):
        """Builds the discriminator module
        The discriminator is a simple feedforward neural network with number of units as specified in `conv_filters`. We have used only linear layers and dropout layers in between. The final layer is a sigmoid layer to ensure that the output is between 0 and 1.
        """
        layers = []
        conv_filters = [self.image_shape[0]] + self.conv_filters

        for i in range(len(conv_filters) - 1):
            # this conv layer preserves the image size
            conv_layer = nn.Conv2d(
                in_channels=conv_filters[i],
                out_channels=conv_filters[i + 1],
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
            )

            # this layer downscales the image
            down_scale_layer = self._get_downscale_layer(
                layer_type=self.downscale_method, channels=conv_filters[i + 1]
            )
            layers.append(conv_layer)
            self.logger.debug(
                f"Added Conv2d layer with in_channels={conv_filters[i]} and out_channels={conv_filters[i+1]}"
            )
            if self.dropout_rates > 0:
                layers.append(nn.Dropout(self.dropout_rates))
                self.logger.debug(f"Added Dropout layer with p={self.dropout_rates}")

            layers.append(self.activation)
            self.logger.debug(f"Added {self.activation} layer")
            layers.append(down_scale_layer)
            self.logger.debug(f"Added downscaling layer")

            # add activation only for conv layers
            if self.downscale_method == "conv":
                layers.append(self.activation)
                self.logger.debug(f"Added {self.activation} layer")

        # add a global max pooling layer to get the output
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.logger.debug(f"Added AdaptiveAvgPool2d layer with output size (1, 1)")
        # flatten and add final layer
        layers.append(nn.Flatten())
        self.logger.debug(f"Added Flatten layer")
        layers.append(nn.Linear(conv_filters[-1], 1))
        self.logger.debug(
            f"Added Linear layer with in_features={conv_filters[-1]} and out_features=1"
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
        return self.model(x)


class DCGAN(GAN):

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        k: int = 1,
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
    ):
        super().__init__(generator, discriminator, k, wandb_run)
