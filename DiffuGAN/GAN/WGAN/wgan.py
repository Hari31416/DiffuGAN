from torch.nn.modules import Module
from DiffuGAN.GAN.networks import BaseGAN
from typing import Union
import torch
import wandb


class WGAN(BaseGAN):

    def __init__(
        self,
        generator: Module,
        discriminator: Module,
        k: int = 5,
        generator_optimizer: str = "rmsprop",
        discriminator_optimizer: str = "rmsprop",
        generator_optimizer_kwargs: dict = {},
        discriminator_optimizer_kwargs: dict = {},
        wandb_run: Union[wandb.sdk.wandb_run.Run, None] = None,
        device: Union[str, None] = None,
        config: dict = {},
        c: float = 0.01,
        **kwargs,
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
            The keyword arguments to pass to the generator optimizer, by default `{}`
        discriminator_optimizer_kwargs : dict, optional
            The keyword arguments to pass to the discriminator optimizer, by default `{}`
        device : str, optional
            The device to use for training, by default None. If None, the device will be set to "cuda" if available, else "cpu"
        wandb_run : Union[wandb.sdk.wandb_run.Run, None], optional
            The wandb run object to log the training, by default None
        config : dict, optional
            The configuration dictionary, by default {}. This will be passed to the wandb run object if it is not None and will be saved locally if the model is saved
        c : float, optional
            The weight clipping value, by default 0.01
        """
        super().__init__(
            generator,
            discriminator,
            k,
            generator_optimizer,
            discriminator_optimizer,
            generator_optimizer_kwargs,
            discriminator_optimizer_kwargs,
            device,
            wandb_run,
            config,
        )
        self.c = c

    def generator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return -torch.mean(self.discriminator(fake))

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        real_part = torch.mean(self.discriminator(real))
        fake_part = torch.mean(self.discriminator(fake))
        return fake_part - real_part

    def after_discriminator_update(self) -> None:
        """Clamp the weights of the discriminator to a small range to enforce the Lipschitz constraint"""
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.c, self.c)
