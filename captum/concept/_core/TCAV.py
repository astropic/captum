#!/usr/bin/env python3

import torch
from torch.nn import Module

from typing import Dict, List, Any, Callable

from captum.concept.fb._core.concept import Concept
from captum.concept.fb._core.concept import ConceptInterpreter
from captum.concept.fb._core.CAV import CAV
from captum.concept.fb._utils.classifier import Classifier
from captum.concept.fb._utils.common import concepts_to_str

from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric

from collections import defaultdict
from functools import reduce

import torch.multiprocessing as multiprocessing


def train_cav(
    concepts,
    layers,
    concept2av_map,
    classifier,
    train_test_split_func,
    save_cavs,
    save_path,
):
    r"""
    Helper function for the actual parallel CAV computing.

    Please see the TCAV class documentation for further information and
    argument description.

    Returns:
        A dictionary of CAV objects indexed by layer name, containing the
        accuracies and the weights of the resulting Linear Model
        training, the respective concepts used for the training and the
        layer name itself.
    """

    concepts_key = concepts_to_str(concepts)
    cavs = defaultdict()
    cavs[concepts_key] = defaultdict()

    for layer in layers:

        # Prepare training set
        x_list = []
        y_list = []
        for i, concept in enumerate(concepts):
            for out in concept2av_map[concept.name][layer]:
                y_list += torch.tensor([i] * out.shape[0])
                x_list += out

        x_train, x_test, y_train, y_test = train_test_split_func(x_list, y_list)

        # Train and evaluate linear model
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        weights = classifier.weights()

        # Linear Classifier accuracy
        idx = y_test == i
        score = y_pred[idx] == y_test[idx]
        accs = score.float().mean()

        cavs[concepts_key][layer] = CAV(
            concepts, layer, {"weights": weights, "accs": accs}, save_path
        )

        if save_cavs:
            cavs[concepts_key][layer].save()

    return cavs


class TCAV(ConceptInterpreter):
    r"""
        Class implementation of the ConceptInterpreter abstract class using
        Tests with Concept Activation Vectors (TCAVs), as in the paper:
            https://arxiv.org/pdf/1711.11279.pdf

        TCAVS are computed by taking the activations of a layer from concept
        tensors and, by using a classifier, a CxF weights tensor (CAV)
        is returned, where:
            F is the layer flattened size
            C is the number of concepts in the classification (#concepts)

        A set of inputs is then used to generate the attributions (ATT) on that
        layer, i.e. the gradients of the output with respect to that layer.
        The CAVs and the ATTs are used to compute the TCAV score:

            0. TCAV = CAV • ATT, a dot product for each input

        Then the TCAVs are used in 2 ways to produce 2 different scores,
        per concept:

            1. sign_count_score = SUM(TCAV) / C, where TCAV > 0, C = #concepts
            2. magnitude_score = SUM(ABS(TCAV * (TCAV > 0))) / SUM(ABS(TCAV))
    """

    def __init__(
        self,
        model: Module,
        concepts: List[Concept],
        layers: List[str],
        classifier: Classifier,
        layer_attribution_func: Callable,
        train_test_split_func: Callable,
        save_path: str = "./cav/",
    ) -> None:
        r"""
        Args:
            model (Module): an instance of pytorch model to extract layer
                        activations and attributions from.
            concepts (list[Concept]): a List of Concept objects to generate the
                        activations from; concept tensors are the instance of
                        the concept "ideas" such as "stripes" in images, or
                        "happy" in text, and are used to create Concept
                        objects, which, in turn, generate the layer
                         activations.
            layers (list[str]): a List of layer names to compute the concept
                        activations and input attributions.
            classifier (Classifier): A custom classifier class, such as the
                        Sklearn "linear_model", that, at a given layer, takes
                        as inputs the layer activations, and targets as the
                        concepts.
            layer_attribution_func (callable): Method to compute the gradients
                        of the output with respect to a given layer
                        (attribution).
            train_test_split_func (callable): Method to split Input and Target
                        data lists into Train and Test data lists.
            save_path (str, optional): Path for saving CAVs.

        Examples::
            >>> # layer_attribution_func:
            >>>
            >>> import captum._utils.gradient as captum_grad
            >>>
            >>> def layer_attributions(model, layer, inputs, target,
            >>>             additional_forward_args):
            >>>
            >>>     attrib, _ , _ =
            >>>         captum_grad.compute_layer_gradients_and_eval(
            >>>                             model,
            >>>                             layer,
            >>>                             inputs, target_ind=target)
            >>>
            >>>     return attrib
            >>>
            >>>
            >>> # train_test_split_func:
            >>>
            >>> def train_test_split(x_list, y_list, test_split=0.33):
            >>>
            >>>     # Shuffle
            >>>     z_list = list(zip(x_list, y_list))
            >>>     random.shuffle(z_list)
            >>>
            >>>     # Split
            >>>     test_size = int(test_split * len(z_list))
            >>>     z_test, z_train = z_list[:test_size], z_list[test_size:]
            >>>     x_test, y_test = zip(*z_test)
            >>>     x_train, y_train = zip(*z_train)
            >>>
            >>>     x_train = torch.stack(x_train)
            >>>     x_test = torch.stack(x_test)
            >>>     y_train = torch.stack(y_train)
            >>>     y_test = torch.stack(y_test)
            >>>
            >>>     return x_train, x_test, y_train, y_test
            >>>
            >>> # TCAV use example:
            >>>
            >>> # "generate_activations" takes input tensors of N concept
            >>> # images Nx3x32x32, and builds a dictionary for all concepts in
            >>> # self.concepts and layers in self.layers,
            >>> # self.concept2av_map[<concept.name>][<layer_name>], containing
            >>> # activation values NxF, where N is the number of concept images
            >>> # and F is the flattened number of features for the layer.
            >>> #
            >>> # "compute_cavs" takes an experimental_set list of N experiments
            >>> # of C Concepts, NxC. Using a classifier, computes (for each
            >>> # experiment Concept tuple) the weights corresponding to each
            >>> # Concept in the tuple. The classifier uses the NxF activation
            >>> # values in concept2av_map (Y) and the respective concepts as
            >>> # labels (X).
            >>> #
            >>> # "interpret" takes an experimental_set (Nx2 Concepts),
            >>> # input images Nx3x32x32 and target=4 (the NN output class
            >>> # label value), and computes the count score and magnitude
            >>> # score for each concept
            >>> # A simple call to the TCAV constructor and to "interpret"
            >>> # invoke all required functions:
            >>> #
            >>> mytcav = TCAV(model=model,
            >>>     # Please check concept.py how to create a Concept object.
            >>>     concepts=concepts,
            >>>     layers=['inception4c', 'inception4d'],
            >>>     # Please check classifier.py how to setup a classifier
            >>>     classifier=classifier,
            >>>     layer_attribution_func=layer_attributions,
            >>>     train_test_split_func=train_test_split)
            >>>
            >>> scores = mytcav.interpret(inputs, experimental_set, target)

            For a more thourough example, please check out TCAV tutorial.
        """

        self.model = model
        self.layers = layers
        self.concepts = concepts
        self.classifier = classifier

        self.concept2av_map = defaultdict()
        self.cavs = defaultdict(lambda: defaultdict())

        self.layer_attribution_func = layer_attribution_func
        self.train_test_split_func = train_test_split_func

        self.save_path = save_path

    def get_module_from_name(self, name: str) -> Any:
        r"""
        Returns the module (layer) object, given its (string) name
        in the model.

        Args:
            name (str): Module or nested modules name string in self.model

        Returns:
            The module (layer) in self.model.
        """

        return reduce(getattr, name.split("."), self.model)

    def generate_all_activations(self):
        r"""
        Computes the layer activations for all Concepts and Layers in,
        respectively, self.layers and self.concepts; concept tensors are the
        instance of the concept "ideas" such as "stripes" in images, or "happy"
        in text, and are used to create Concept objects, which, in turn,
        generate the layer activations.
        """

        for concept in self.concepts:
            self.generate_activation(self.layers, concept)

    def generate_activation(self, layers: List[str], concept: Concept):
        r"""
        Computes the layer activations for the specified concept and
        list of layers; concept tensors are the instance of the concept
        "ideas" such as "stripes" in images, or "happy"
        in text, and are used to create Concept objects, which, in turn,
        generate the layer activations.

        Args:
            layers (list[str]): a List of layer names to compute the
                        activations at, w.r.t. given Concept.
            concept (Concept): a single Concept object; concept tensors are the
                        instance of the concept "ideas" such as "stripes" in
                        images, or "happy" in text, and are used to create
                        Concept objects, which, in turn, generate the layer
                        activations.
        """

        if concept.name not in self.concept2av_map:
            self.concept2av_map[concept.name] = defaultdict(list)

        def forward_hook_wrapper(layer_name):
            def forward_hook(module, inp, out=None):
                out = torch.reshape(out, (out.shape[0], -1))
                # TODO: it would be better to use an iterator here (T69119305)
                # out.shape = NxF, N=concept batch size, F = feature size
                self.concept2av_map[concept.name][layer_name].append(out.detach())

            return forward_hook

        hooks = []
        for layer in layers:
            layer_module = self.get_module_from_name(layer)
            hooks.append(
                layer_module.register_forward_hook(forward_hook_wrapper(layer))
            )

        for examples in concept.data_iter:
            self.model(examples)

        for hook in hooks:
            hook.remove()

    def generate_activations(self, concept_layers: Dict[Concept, List[str]]):
        r"""
        Computes the activations for the specified concept and
        list of layers pairs; concept tensors are the instance of the concept
        "ideas" such as "stripes" in images, or "happy" in text, and are used
        to create Concept objects, which, in turn, generate the layer
        activations.

        Args:
            concept_layers (dict[Concept, list[str]]): Dictionay that maps
                        Concept objects to a list of layer names to generate
                        the activations. Ex.: concept_layers =
                        {"striped": ['inception4c', 'inception4d']}
        """

        for concept in concept_layers:
            self.generate_activation(concept_layers[concept], concept)

    def load_cavs(self, concepts: List[Concept]):
        r"""
        Function to load a dictionary of CAVs, present in
        self.save_path, in .pkl files with the format:
        <self.save_path><concepts key>-<layer name>.pkl. Ex.:
        "/cavs/striped-random_0-random_1-inception4c.pkl"

        Returns a dictionay with unloaded (Concepts, layers) CAV values,
        mapping each Concept object to the corresponding List of layer names,
        and a List of all layer names present in the dictionary.

        Args:
            concepts (list[Concept]): List of Concept objects to load the CAVs
                        for, with all layers in self.layers.

        Returns:
            layers (list[layer]): A list of layers not loaded, contained in
                concept_layers, for convenience.
            concept_layers (dict[concept, layer]): A dictionay of unloaded maps
                of Concept objects to a List of layer names.
        """

        concepts_key = concepts_to_str(concepts)

        layers = []
        concept_layers = defaultdict(list)

        for layer in self.layers:

            self.cavs[concepts_key][layer] = CAV.load(self.save_path, concepts, layer)

            # If CAV not loaded, or force_train, train
            if (
                concepts_key not in self.cavs
                or layer not in self.cavs[concepts_key]
                or not self.cavs[concepts_key][layer]
            ):

                layers.append(layer)
                # For all concepts in this experimental_set
                for concept in concepts:

                    # Collect not activated layers for this concept
                    if (
                        concept.name not in self.concept2av_map
                        or layer not in self.concept2av_map[concept.name]
                        or not self.concept2av_map[concept.name][layer]
                    ):

                        concept_layers[concept].append(layer)

        return layers, concept_layers

    def compute_cavs(
        self,
        experimental_set: List[List[Concept]],
        save_cavs: bool = True,
        force_train: bool = False,
        run_parallel: bool = False,
        processes: int = None,
    ):
        r"""
        Compute the CAVs, e.g. run the classifier training and, by default,
        save the resulting weights from the training. The weights have
        a CxF size, where C is the number of concepts and F is the layer
        flattened size.

        Args:
            experimental_set (list[list[Concept]]): A list of lists of concept
                        tuples.
            save_cavs (bool): Whether or not to save the CAV's.
            force_train (bool): Retrain the CAV's regardless they are
                        saved or not.
            run_parallel (bool): Whether or not to compute each CAV
                        experimental set in parallel (in diferent cores)
                        or not.
            processes (int, optional): The number of processes to be created if
                    run_parallel = True. If not specified, the system will
                    take up as many cores as available.

        Returns:
            A dictionary of CAV objects indexed by layer name, containing the
            accuracies and the weights of the resulting Linear Model
            training, the respective concepts used for the training and the
            layer name itself.
        """

        # Update self.concepts
        concept_set = set(self.concepts)
        for concepts in experimental_set:
            concept_set.update(concepts)
        self.concepts = list(concept_set)

        if force_train:
            self.generate_all_activations()

        # List of layers per concept key (experimental_set item) to be trained
        concepts_layers = defaultdict(list)

        for concepts in experimental_set:

            concepts_key = concepts_to_str(concepts)

            # If not 'force_train', try to load a saved CAV
            if not force_train:

                layers, concept_layers = self.load_cavs(concepts)
                concepts_layers[concepts_key] = layers

                # Generate activations for missing (concept, layers)
                self.generate_activations(concept_layers)

            else:
                concepts_layers[concepts_key] = self.layers

        if run_parallel:

            pool = multiprocessing.Pool(processes)
            cavs_list = pool.starmap(
                train_cav,
                [
                    (
                        concepts,
                        concepts_layers[concepts_to_str(concepts)],
                        self.concept2av_map,
                        self.classifier,
                        self.train_test_split_func,
                        save_cavs,
                        self.save_path,
                    )
                    for concepts in experimental_set
                ],
            )
            pool.close()
            pool.join()

        else:

            cavs_list = []
            for concepts in experimental_set:
                cavs_list.append(
                    train_cav(
                        concepts,
                        concepts_layers[concepts_key],
                        self.concept2av_map,
                        self.classifier,
                        self.train_test_split_func,
                        save_cavs,
                        self.save_path,
                    )
                )

        # list[Dict[concept, Dict[layer, list]]] => Dict[concept, Dict[layer, list]]
        for cavs in cavs_list:
            for c_key in cavs:
                self.cavs[c_key].update(cavs[c_key])

        return self.cavs

    def interpret(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        experimental_set: List[List[Concept]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        run_parallel: bool = False,
        processes: int = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        r"""
        Computes the TCAVs and TCAV scores: Compute the layer attributions, e.g.
        gradients of the output with respect to the layers and "compares" them
        with the classifier computed weights (CAVs) through a dot product
        (tcav_score).

        Args:
            inputs (tensor or tuple of tensors): The tensors do be analyzed
                        w.r.t. the concepts. Input for which attributions are
                        computed. If forward_func takes a single tensor as
                        input, a single input tensor should be provided.
                        If layer_attribution_func takes multiple tensors as
                        input, a tuple of the input tensors should be provided.
                        It is assumed that for all given input tensors,
                        dimension 0 corresponds to the number of examples
                        (aka batch size), and if multiple input tensors are
                        provided, the examples must be aligned appropriately.
            experimental_set (list[list[Concept]]): A list of Concept tuples.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:
                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples
                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.
                        For outputs with > 2 dimensions, targets can be either:
                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.
                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.
            additional_forward_args (any): Extra arguments for computing the
                        attributions, if needed.
            run_parallel (bool): Whether or not to compute each CAV
                        experimental set in parallel (in diferent cores)
                        or not.
            processes (int, optional): The number of processes to be created if
                        run_parallel = True. If not specified, the system will
                        take up as many cores as available.

        Returns:
            A dictionary of count and magitude scores per layer, per concept.

        Score output example::
            >>> #
            >>> # scores =
            >>> # {'striped-random1':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.5800, 0.4200]),
            >>> #          'magnitude': tensor([0.6613, 0.3387])},
            >>> #      'inception4d':
            >>> #         k{'sign_count': tensor([0.6200, 0.3800]),
            >>> #           'magnitude': tensor([0.7707, 0.2293])}}),
            >>> #  'striped-random2':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.6200, 0.3800]),
            >>> #          'magnitude': tensor([0.6806, 0.3194])},
            >>> #      'inception4d':
            >>> #         {'sign_count': tensor([0.6400, 0.3600]),
            >>> #          'magnitude': tensor([0.6563, 0.3437])}})})
            >>> #

        """

        self.compute_cavs(experimental_set, run_parallel)

        scores = defaultdict(lambda: defaultdict())

        for layer in self.layers:

            layer_module = self.get_module_from_name(layer)

            # TODO: layer_attribution_func could be exposed as a regular
            # available 'attribution' method (T69438880)
            attrib = self.layer_attribution_func(
                self.model, layer_module, inputs, target, additional_forward_args
            )

            # Temporary block while 'layer_attribution_func' exists, until it
            # is replaced by a regular 'attribute' method.
            if type(attrib) is tuple:
                attrib = attrib[0]
            sh = attrib.shape
            # n_inputs x n_features (2 dimensions)
            attrib = torch.reshape(attrib, (sh[0], -1))
            assert (
                len(attrib.shape) == 2
            ), "attrib should have 2 dimensions: n_inputs x n_features."

            # n_experiments x n_concepts x n_features (3 dimensions)
            cav = []
            for concepts in experimental_set:
                concepts_key = concepts_to_str(concepts)
                cav.append(self.cavs[concepts_key][layer].stats["weights"].float())
            cav = torch.stack(cav)
            assert (
                len(cav.shape) == 3
            ), "cav should have 3 dimensions: n_experiments x n_concepts x n_features."

            # n_inputs x n_concepts (2 dimensions)
            tcav_score = torch.matmul(attrib.float(), torch.transpose(cav, 1, 2))
            assert (
                len(tcav_score.shape) == 3
            ), "tcav_score should have 3 dimensions: n_experiments x n_inputs x n_concepts."

            assert (
                attrib.shape[0] == tcav_score.shape[1]
            ), "attrib and tcav_score should have the same 1st and 2nd dimensions respectively (n_inputs)."

            # n_experiments x n_concepts
            sign_count_score = (
                torch.sum(tcav_score > 0.0, dim=1).float() / tcav_score.shape[1]
            )

            magnitude_score = torch.sum(
                torch.abs(tcav_score * (tcav_score > 0.0)), dim=1
            ) / torch.sum(torch.abs(tcav_score), dim=1)

            for i, concepts in enumerate(experimental_set):
                concepts_key = concepts_to_str(concepts)
                scores[concepts_key][layer] = {
                    "sign_count": sign_count_score[i, :],
                    "magnitude": magnitude_score[i, :],
                }

        return scores
