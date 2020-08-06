#!/usr/bin/env python3

from typing import Dict, List

from captum.concept.fb._core.concept import Concept
import torch
import os


class CAV:
    r"""
    A Concept Activation Vector (CAV) is a vector orthogonal to the decision
    boundary provided by a classifier to distinguish between activations
    produced by concept’s examples and random counter concept examples, on one
    layer, as described in the paper:
        https://arxiv.org/pdf/1711.11279.pdf
    """

    def __init__(
        self,
        concepts: List[Concept],
        layer: str,
        stats: Dict = None,
        save_path: str = "./cav/",
    ) -> None:
        r"""
        This class encapsulates the instances of CAVs objects, saves them in
        and loads them from the manifold.

        Args:
            concepts (list[Concept]): a List of Concept objects. Only their
                        names will be saved and loaded.
            layer (str): The layer where concept activation vectors are
                        computed using a predefined classifier.
            stats (dict): a dictionary that retains information about the CAV
                        classifier such as CAV weights and accuracies.
                        Ex.: stats = {"weights": weights, "accs": accs}, where
                        "weights" is the classifier result and "accs" the
                        classifier training accuracy.
            save_path (str): the manifold path where the CAV objects are saved.

        """

        self.concepts = concepts
        self.layer = layer
        self.stats = stats
        self.save_path = save_path

    @staticmethod
    def assemble_save_path(path: str, concepts: List[str], layer: str):
        r"""
        Utility method for assembling filename and its path, from
        a concept list and a layer name.

        Args:
            path (str): a path to be concatenated with the concepts key and
                        layer name.
            concepts (list): a List of concept names to be concatenated and
                        used as a concepts key. These concept names are
                        respective to the Concept objects used for the
                        classifier train.
            layer (str): the layer name where the activations are
                        computed for.

        Returns:
            A string containing the path where the computed CAVs will be saved.
            For example, given:
                concepts = ["striped", "random_0", "random_1"]
                layer = "inception4c"
                path = "/cavs",
            the resulting save path will be:
                "/cavs/striped-random_0-random_1-inception4c.pkl"

        """

        file_name = "-".join([c.name for c in concepts]) + "-" + layer + ".pkl"

        return os.path.join(path, file_name)

    def save(self):
        r"""
        Saves a dictionary of the CAV computed values into a pickle file at the
        location returned by the "assemble_save_path" static function. This
        dictionary contains the concept names list, the layer name where the
        activations are computed for, the stats dictionary that contains
        information about the classifier train result such as the weights
        and training accuracies, and the path where the CAV computed values
        will be saved. Ex.:

        save_dict = {
            "concept_names": ["striped", "random_0", "random_1"],
            "layer": "inception4c",
            "stats": {"weights": weights, "accs": accs},
            "save_path": "/cavs/striped-random_0-random_1-inception4c.pkl"
        }

        """

        save_dict = {
            "concept_names": [c.name for c in self.concepts],
            "layer": self.layer,
            "stats": self.stats,
            "save_path": self.save_path,
        }

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        cavs_path = CAV.assemble_save_path(self.save_path, self.concepts, self.layer)

        torch.save(save_dict, cavs_path)

    @staticmethod
    def load(cavs_path: str, concepts: List[str], layer: str):
        r"""
        Loads a CAV dictionary from a pickle file.

        Args:
            cavs_path (str): the path where to load the CAV corresponding to
                        the desired input concepts and layer.
                        Ex.: "/cavs"
            concepts (list[str]):  a List of concept names.
                        Ex.: ["striped", "random_0", "random_1"]
            layer (str): the layer name. Ex.: "inception4c". In case of nested
                        layers we use dots to specify the depth / hierarchy.
                        Ex.: "layer.sublayer.subsublayer"

        Returns:
            An instance of a CAV class, containing the respective CAV scores
            per concept, from a path assembled from the arguments as, for
            example: "/cavs/striped-random_0-random_1-inception4c.pkl"
        """

        cavs_path = CAV.assemble_save_path(cavs_path, concepts, layer)

        if os.path.exists(cavs_path):
            save_dict = torch.load(cavs_path)

            concept_names = save_dict["concept_names"]
            concepts = [Concept(i, c, None) for i, c in enumerate(concept_names)]
            cav = CAV(concepts, save_dict["layer"], save_dict["stats"])

            return cav

        return None
