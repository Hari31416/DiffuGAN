import torch


def w_distance_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Wasserstein distance loss

    Parameters
    ----------
    fake : torch.Tensor
        Fake image
    real : torch.Tensor
        Real image
    criterion : torch.nn.Module
        Loss function
    """
    return -torch.mean(real) + torch.mean(fake)


def discriminator_w_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Loss for discriminator

    Parameters
    ----------
    fake : torch.Tensor
        Fake image
    real : torch.Tensor
        Real image
    criterion : torch.nn.Module
        Loss function
    """
    return -torch.mean(real) + torch.mean(fake)


def discriminator_loss(
    fake: torch.Tensor, real: torch.Tensor, criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Loss for discriminator

    Parameters
    ----------
    fake : torch.Tensor
        Fake image
    real : torch.Tensor
        Real image
    criterion : torch.nn.Module
        Loss function
    """
    fake_loss = criterion(fake, torch.zeros_like(fake))
    real_loss = criterion(real, torch.ones_like(real))
    return (fake_loss + real_loss) / 2


def generator_og_loss(
    fake: torch.Tensor, real: torch.Tensor, criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Original GAN loss for generator

    Parameters
    ----------
    fake : torch.Tensor
        Fake image
    real : torch.Tensor
        Real image
    criterion : torch.nn.Module
        Loss function
    """
    return criterion(fake, real)
