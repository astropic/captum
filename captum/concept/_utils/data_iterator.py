#!/usr/bin/env python3

import glob

from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Dataset

from typing import List, Callable


class CustomIterableDataset(IterableDataset):
    r"""
        Auxiliary class for iterating through a dataset.
    """

    def __init__(self, transform_filename_to_tensor: Callable, path: str) -> None:
        r"""
        Args:
            transform_filename_to_tensor (callable): Function to read a data
                        file from path and return a tensor from that file.
            path (str): Path to dataset files
        """

        self.path = path
        self.file_itr = glob.glob(self.path + "*")
        self.transform_filename_to_tensor = transform_filename_to_tensor

    def __iter__(self) -> map([str, Tensor], List[str]):
        r"""
        Returns:
            A map from a function that processes a list of paths to
            a list of Tensors.
        """

        return map(self.transform_filename_to_tensor, self.file_itr)


def get_concept_iterator_from_path(
    dataset: Dataset, batch_size: int = 64
) -> DataLoader:
    r"""
    Auxiliary function to create a torch DataLoader from a path containing
    files with concepts and a function to read the files in that path.

    Args:
        dataset (Dataset): A Dataset Torch object to provide the data to
                    iterate from.
        batch_size (int, optional): Batch size of concept tensors.

    Returns:
        dataloader_iter: a DataLoader for data iteration.
    """

    dataloader_iter = DataLoader(dataset, batch_size=batch_size)

    return dataloader_iter
