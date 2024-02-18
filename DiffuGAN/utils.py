"""Some utility functions/classes to be used throughout the project."""

from .env import env

import logging
import argparse
import inspect
from typing import Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb


def create_simple_logger(
    logger_name: str, level: str = env.LOG_LEVEL
) -> logging.Logger:
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


def get_function_arguments(func: callable) -> list[str]:
    """Returns the names of the arguments of the given function.

    Parameters
    ----------
    func : callable
        The function.

    Returns
    -------
    list[str]
        The names of the arguments of the function.
    """
    return inspect.getfullargspec(func).args


def update_kwargs_for_function(
    func: callable, kwargs: dict[str, any], raise_error: bool = False
) -> dict[str, any]:
    """Updates the given keyword arguments to include only those that are supported by the function.

    Parameters
    ----------
    func : callable
        The function.
    kwargs : dict[str, any]
        The keyword arguments to be updated.
    raise_error : bool, optional
        Whether to raise an error if an unsupported argument is found. Default is False.

    Returns
    -------
    dict[str, any]
        The updated keyword arguments.
    """
    supported_args = get_function_arguments(func)
    logger.debug(
        f"Supported arguments for the function {func.__name__}: {supported_args}"
    )
    all_args = list(kwargs.keys())
    for arg in all_args:
        if arg not in supported_args:
            msg = f"Argument {arg} is not supported by the function {func.__name__}."
            if raise_error:
                logger.error(msg)
                raise ValueError(msg)
            logger.info(msg)
            kwargs.pop(arg)
    return kwargs


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


class ScaleTransform:
    def __init__(self, scale_type: str = "0-1") -> None:
        self.scale_type = scale_type

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        min_val = torch.min(img)
        max_val = torch.max(img)
        if self.scale_type == "0-1":
            img = (img - min_val) / (max_val - min_val)
        elif self.scale_type == "-1-1":
            img = 2 * (img - min_val) / (max_val - min_val) - 1
        else:
            raise ValueError("Invalid scale type. Must be '0-1' or '-1-1'")
        return img


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


def create_grid_from_batched_image(
    images: Union[torch.Tensor, np.ndarray],
    nrow: Union[int, None] = None,
    padding: int = 2,
    pad_value: int = 0,
    normalize: bool = True,
    return_type: str = "numpy",
    quick_plot: bool = False,
) -> Union[torch.Tensor, np.ndarray, Image.Image]:
    """Creates a grid of images from the given batch of images.

    Parameters
    ----------
    images : torch.Tensor | np.ndarray
        The batch of images to be displayed. Must be in the shape (N, H, W, C).
    nrow : int | None, optional
        Number of images in each row. If None, it is set to the square root of the number of images. Default is None.
    padding : int, optional
        The padding between the images. Default is 2.
    pad_value : int, optional
        The value to be used for padding. Default is 0.
    normalize : bool, optional
        Whether to normalize the images. Default is True.
    return_type : str, optional
        The type of the returned grid. Must be one of "numpy" "tensor" or "PIL". Default is "numpy".
    quick_plot : bool, optional
        Plots the image using matplotlib if True. Default is False.

    Returns
    -------
    np.ndarray
        The grid of images.
    """
    # if the input is a tensor, convert it to numpy
    if torch.is_tensor(images):
        images = images.numpy()

    num_images, height, width, channels = images.shape
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(num_images)))
    ncols = int(np.ceil(images.shape[0] / nrow))

    grid = np.ones((nrow * (width + padding), ncols * (width + padding), channels))
    grid *= pad_value

    # fill the array with the images
    for i in range(nrow):
        for j in range(ncols):
            index = i * ncols + j
            if index < num_images:
                grid[
                    i * (height + padding) : i * (height + padding) + height,
                    j * (width + padding) : j * (width + padding) + width,
                    :,
                ] = images[index]

    if normalize:
        grid -= grid.min()
        grid /= grid.max()

    if return_type == "tensor":
        grid = torch.from_numpy(grid)
    elif return_type == "PIL":
        # make sure the grid is in the range [0, 255]
        max_val = grid.max()
        if max_val <= 1:
            grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
    if quick_plot:
        show_image(grid, title="Grid of images")
    return grid


def show_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    title: str = "",
    ax: Union[plt.Axes, None] = None,
    cmap: str = "viridis",
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
    channels = image.shape[-1]

    if channels == 1 and cmap not in ["gray", "Greys"]:
        cmap = "gray"
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.show()
    return ax


def parse_activation(name: str, **kwargs: dict[str, Any]) -> torch.nn.Module:
    """Parses the activation function name and returns the corresponding function.

    Parameters
    ----------
    name : str
        Name of the activation function.
    kwargs
        Additional keyword arguments to be passed to the activation function.
        **Note**: Use `inplace=True` if using the activations in a `torch.nn.Sequential` block.

    Returns
    -------
    torch.nn.Module
        The activation function.
    """
    name = name.lower()
    str_to_activation_map = {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softmin": torch.nn.Softmin,
    }
    if name not in str_to_activation_map:
        msg = f"Activation function {name} is not supported. Must be one of {list(str_to_activation_map.keys())}."
        logger.error(msg)
        raise NotImplementedError(msg)
    # make sure that only those keyword arguments that are supported by the activation function are passed
    activation = str_to_activation_map[name]
    supported_kwargs = update_kwargs_for_function(activation, kwargs, raise_error=False)
    return activation(**supported_kwargs)


def parse_optimizer(
    name: str, parameters: torch.nn.Module, **kwargs: dict[str, Any]
) -> torch.optim.Optimizer:
    """Parses the optimizer name and returns the corresponding optimizer.

    Parameters
    ----------
    name : str
        Name of the optimizer.
    parameters: torch.nn.Module
        The parameters to be optimized.
    kwargs
        Additional keyword arguments to be passed to the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    """
    name = name.lower()
    str_to_optimizer_map = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    if name not in str_to_optimizer_map:
        msg = f"Optimizer {name} is not supported. Must be one of {list(str_to_optimizer_map.keys())}."
        logger.error(msg)
        raise NotImplementedError(msg)
    # make sure that only those keyword arguments that are supported by the optimizer are passed
    optimizer = str_to_optimizer_map[name]
    supported_kwargs = update_kwargs_for_function(optimizer, kwargs, raise_error=False)
    return optimizer(parameters, **supported_kwargs)


def create_wandb_logger(
    name: Union[str, None] = None,
    project: Union[str, None] = None,
    config: Union[dict[str, any], None] = None,
    tags: Union[list[str], None] = None,
    notes: str = "",
    group: Union[str, None] = None,
    job_type: str = "",
    logger: Union[logging.Logger, None] = None,
) -> wandb.sdk.wandb_run.Run:
    """Creates a new run on Weights & Biases and returns the run object.

    Parameters
    ----------
    project : str | None, optional
        The name of the project. If None, it must be provided in the config. Default is None.
    name : str | None, optional
        The name of the run. If None, it must be provided in the config. Default is None.
    config : dict[str, any] | None, optional
        The configuration to be logged. Default is None. If `project` and `name` are not provided, they must be present in the config.
    tags : list[str] | None, optional
        The tags to be added to the run. Default is None.
    notes : str, optional
        The notes to be added to the run. Default is "".
    group : str | None, optional
        The name of the group to which the run belongs. Default is None.
    job_type : str, optional
        The type of job. Default is "train".
    logger : logging.Logger | None, optional
        The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.

    Returns
    -------
    wandb.Run
        The run object.
    """
    logger = logger or create_simple_logger("create_wandb_logger")
    if config is None:
        logger.debug("No config provided. Using an empty config.")
        config = {}

    if name is None and "name" not in config.keys():
        m = "Run name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    if project is None and "project" not in config.keys():
        m = "Project name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    # If the arguments are provided, they take precedence over the config
    name = name or config.get("name")
    project = project or config.get("project")
    notes = notes or config.get("notes")
    tags = tags or config.get("tags")
    group = group or config.get("group")
    job_type = job_type or config.get("job_type")

    logger.info(
        f"Initializing Weights & Biases for project {project} with run name {name}."
    )
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
    )
    return wandb


def yaml_text_to_dict(yaml_text: str) -> dict[str, any]:
    """Converts the given YAML text to a dictionary.

    Parameters
    ----------
    yaml_text : str
        The YAML text.

    Returns
    -------
    dict[str, any]
        The dictionary representation of the YAML text.
    """
    import yaml

    return yaml.safe_load(yaml_text)


def add_dataset_args(
    args: argparse.ArgumentParser, default_arguments: dict[str, any] = {}
):
    """Adds arguments related to the dataset to the given parser."""
    # dataset parameters
    args.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        default=default_arguments.get("dataset_name", "mnist"),
        choices=["mnist", "fashionmnist", "cifar10", "celeba"],
        help="The dataset to use.",
    )
    args.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=default_arguments.get("batch_size", 32),
        help="The batch size.",
    )
    args.add_argument(
        "--root",
        "-r",
        type=str,
        default=default_arguments.get("root", "data"),
        help="The root directory for the dataset.",
    )
    args.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=default_arguments.get("num_workers", 1),
        help="The number of workers.",
    )
    args.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=default_arguments.get("shuffle", True),
        help="Shuffle the dataset.",
    )
    args.add_argument(
        "--split-train",
        action=argparse.BooleanOptionalAction,
        default=default_arguments.get("split_train", True),
        help="Whether to use the training set.",
    )
    logger.debug("Added dataset arguments to the parser.")
    return args


def add_wandb_args(
    args: argparse.ArgumentParser, default_arguments: dict[str, any] = {}
):
    """Adds arguments related to Weights & Biases to the given parser."""
    args.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use Weights & Biases.",
    )
    args.add_argument(
        "--project",
        type=str,
        default=default_arguments.get("project", "diffugan"),
        help="The name of the Weights & Biases project.",
    )
    args.add_argument(
        "--run",
        type=str,
        default=default_arguments.get("name", "run"),
        help="The name of the Weights & Biases run.",
    )
    args.add_argument(
        "--notes",
        type=str,
        default=default_arguments.get("notes", ""),
        help="The notes to be added to the Weights & Biases run.",
    )
    args.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=default_arguments.get("tags", []),
        help="The tags to be added to the Weights & Biases run.",
    )
    args.add_argument(
        "--group",
        type=str,
        default=default_arguments.get("group", ""),
        help="The name of the group to which the Weights & Biases run belongs.",
    )
    logger.debug("Added Weights & Biases arguments to the parser.")
    return args
