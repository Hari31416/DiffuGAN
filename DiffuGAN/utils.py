"""Some utility functions/classes to be used throughout the project."""

import DiffuGAN.env as env

import logging
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_simple_logger(logger_name: str, level: str = "info") -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical".

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = create_simple_logger("utils", env.LOG_LEVEL)


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


class ImageDataset(Dataset):
    """A class that fetches some built-in datasets and provides a simple interface to access them. The class also provides methods to transform the data and to create a DataLoader."""

    def __init__(self, logger: Union[logging.Logger, None] = None) -> None:
        """Initializes the ImageDataset object.

        Parameters
        ----------
        logger : logging.Logger | None, optional
            The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.
        """
        self.logger = logger or create_simple_logger("ImageDataset")

    def _load_dataset(
        self,
        dataset_name: str,
        root: str,
        transform: Union[torch.nn.Module, None] = None,
        train: bool = True,
        **kwargs: dict[str, any],
    ) -> torch.utils.data.Dataset:
        """Loads the dataset with the given name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to be loaded. Currently, only "mnist" is supported.
        root : str
            The root directory where the dataset is stored.
        transform : torch.nn.Module
            The transformation to be applied to the data.
        train : bool
            Whether to load the training data.

            - If True, only the training data is loaded.
            - If False, only the test data is loaded.
            - If None, both the training and test data are loaded.

        kwargs
            Additional keyword arguments to be passed to the dataset class.

        Returns
        -------
        torch.utils.data.Dataset
            The dataset object.
        """
        if transform is None:
            self.logger.info(
                "No transformation provided. Using the default transformation, `ToTensor`."
            )
            transform = transforms.ToTensor()

        dataset_name = dataset_name.lower()
        dataset_name_to_class_map = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
            "celeba": datasets.CelebA,
        }
        if dataset_name not in dataset_name_to_class_map:
            msg = f"Dataset {dataset_name} is not supported. Must be one of {list(dataset_name_to_class_map.keys())}."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        # add train parameter to kwargs
        kwargs["train"] = train
        # cleba has split as a parameter
        if dataset_name == "celeba":
            self.logger.info(
                "CelebA dataset has a 'split' parameter instead of 'train'. Using 'split' instead. If `train` is True, `split` will be 'train'. Otherwise, it will be 'test'."
            )
            kwargs["split"] = "train" if train else "test"
            # remove train from kwargs
            kwargs.pop("train")

        dataset = dataset_name_to_class_map[dataset_name](
            root=root, transform=transform, download=True, **kwargs
        )
        return dataset

    def load_dataset(
        self,
        dataset_name: str,
        batch_size: int,
        root: str = "data",
        transform: Union[torch.nn.Module, None] = None,
        train: bool = True,
        shuffle: bool = True,
        num_workers: int = 1,
        **kwargs: dict[str, any],
    ) -> DataLoader:
        """Loads the dataset with the given name and creates a DataLoader for it.


        Parameters
        ----------
        dataset_name : str
            Name of the dataset to be loaded.
        batch_size : int
            The batch size.
        transform : torch.nn.Module
            The transformation to be applied to the data.
        train : bool
            Whether to load the training data.

            - If True, only the training data is loaded.
            - If False, only the test data is loaded.
            - If None, both the training and test data are loaded.

        shuffle : bool
            Whether to shuffle the data.
        num_workers : int
            Number of workers to use for loading the data.
        kwargs
            Additional keyword arguments to be passed to the dataset class.

        Returns
        -------
        torch.utils.data.DataLoader or tuple
            If `train` is None, a tuple of two DataLoader objects is returned, one for the training data and one for the test data. Otherwise, a single DataLoader object is returned.
        """
        if train is None:
            self.logger.info(
                "`train` is set to None. Loading both train and test datasets."
            )
            # Meaning that both train and test data should be loaded
            train_dataset = self._load_dataset(
                dataset_name, root, transform, train=True, **kwargs
            )
            test_dataset = self._load_dataset(
                dataset_name, root, transform, train=False, **kwargs
            )
            return (
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                ),
                DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                ),
            )

        self.logger.info(f"Loading {dataset_name} dataset.")
        dataset = self._load_dataset(
            dataset_name, root, transform, train=train, **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


def show_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    title: str = "",
    ax: Union[plt.Axes, None] = None,
) -> plt.Axes:
    """Displays the given image.

    Parameters
    ----------
    image : torch.Tensor
        The image to be displayed.
    title : str, optional
        The title of the image. Default is "".
    ax : plt.Axes | None, optional
        The axes on which to display the image. If None, a new figure is created. Default is None.

    Returns
    -------
    plt.Axes
        The axes on which the image is displayed.
    """
    # TODO: Update the method to handle more complex cases
    if ax is None:
        logger.debug("Creating a new figure as no axes were provided.")
        _, ax = plt.subplots()

    if isinstance(image, torch.Tensor):
        logger.debug("Converting tensor to numpy and permuting the dimensions.")
        # convert to numpy and permute the dimensions
        image = image.permute(1, 2, 0).numpy()
    # If the image is a PIL image, convert it to numpy
    if isinstance(image, Image.Image):
        logger.debug("Converting PIL image to numpy.")
        image = np.array(image)

    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    plt.show()
    return ax
