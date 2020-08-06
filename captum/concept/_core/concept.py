#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor
from torch.nn import Module


class Concept:

    r"""
    Concept features are human-friendly idea representations of target classes
    in a Machine Learning task, in a tensor format: they can be images such as
    "stripes" images for the zebra image target class in Computer Vision, the
    word "happy" for the "positive" sentiment target class in NLP, or a
    particular shape in the data for a forecast task. This class contains
    a concept features iterator as described in the paper:
        https://arxiv.org/pdf/1711.11279.pdf
    """

    def __init__(self, id: int, name: str, data_iter: iter) -> None:

        r"""
        Args:
            id (int): Identifier. It can be any unique identifier integer per
                        Concept object.
            name (str): Concept name. It can be any unique identifier integer
                        per Concept object. It can be the same folder name that
                        contains the concept files, such as images. For example:
                        "striped", "random", "random_0", "random_1".
            data_iter (iter): A pytorch Dataloader object. Combines a dataset
                        and a sampler, and provides an iterable over a given
                        dataset. For more information, please check:
                        https://pytorch.org/docs/stable/data.html

        Example::
            >>> # Creates a Concept object named "striped", with a data_iter
            >>> # object to iterate over all files in "./concepts/striped"
            >>> concept_name = "striped"
            >>> concept_path = os.path.join("./concepts", concept_name) + "/"
            >>> concept_iter = get_concept_iterator_from_path(
            >>> get_tensor_from_filename, concepts_path=concept_path)
            >>> concept_object = Concept(
                    id=0, name=concept_name, data_iter=concept_iter)
        """

        self.id = id
        self.name = name
        self.data_iter = data_iter


class ConceptInterpreter(ABC):
    r"""
        Abstract Class for the implementation of the
        Tests with Concept Activation Vectors (TCAVs),
        as in the paper:
            https://arxiv.org/pdf/1711.11279.pdf
        A concept interpreter class must receive a model and input tensors
        to be analyzed w.r.t. the concepts.
    """

    @abstractmethod
    def __init__(self, model: Module) -> None:

        pass

    @abstractmethod
    def interpret(self, inputs: Tensor, additional_forward_args: Any = None):

        pass
