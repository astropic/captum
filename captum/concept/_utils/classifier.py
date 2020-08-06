#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List
from torch import Tensor


class Classifier(ABC):
    r"""
        Abstract Class for the implementation of a classifier
        (in principle a linear classifier) for the Tests with
        Concept Activation Vectors (TCAVs), as in the paper:
            https://arxiv.org/pdf/1711.11279.pdf

        The classifier will be trained on activations of particular layer,
        activations computed by Concept inputs and matching Concept labels as
        outputs.

        Example::

        >>> from sklearn import linear_model
        >>>
        >>> class CustomClassifier(Classifier):
        >>>
        >>> def __init__(self):
        >>>
        >>>     self.lm = linear_model.SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        >>>
        >>> def fit(self, inputs, labels):
        >>>
        >>>     self.lm.fit(inputs.detach().numpy(), labels.detach().numpy())
        >>>
        >>> def predict(self, inputs):
        >>>
        >>>     return torch.tensor(self.lm.predict(inputs.detach().numpy()))
        >>>
        >>> def weights(self):
        >>>
        >>>     if len(self.lm.coef_) == 1:
        >>>         # if there are two concepts, there is only one label. We split it in two.
        >>>         return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
        >>>     else:
        >>>         return torch.tensor(self.lm.coef_)


    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, inputs: Tensor, labels: List[int]) -> None:
        r"""
        This function is responsible for training the classifier from
        a NxF Tensor (layer activations from concept tensor inputs) and a list
        of N int that matches the size of the Tensor first dimension. F is the
        flattened tensor layer feature size.

        Args:
            inputs (tensor): a Torch Tensor. This tensor is a set of layer
                        activations.
            labels (list(int)): a list of int that matches the Tensor first
                        dimension. These labels are concept identifiers.
        """
        pass

    @abstractmethod
    def predict(self, inputs: Tensor) -> Tensor:
        r"""
        This function returns the predicted output values from the
        input tensor (concept label predictions from layer activations).

        Args:
            inputs (tensor): a Torch Tensor, representing a set of layer
                        activations.

        Returns:
            A Torch Tensor, indicating the predicted Concept labels.
        """
        pass

    @abstractmethod
    def weights(self) -> Tensor:
        r"""
        This function returns a CxF tensor weights, where N is the number of
        inputs, C the number of classes (Concepts) and F is the flattened
        tensor layer feature size.

        Returns:
            A torch Tensor with the weights resulting from the Mmdel training.
            The model is traditionally a linear model suggested by the
            original paper.
        """
        pass
