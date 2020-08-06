#!/usr/bin/env python3import

import os
import glob
import tempfile
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

import captum._utils.gradient as captum_grad

from captum.concept.fb._core.concept import Concept
from captum.concept.fb._core.TCAV import TCAV
from captum.concept.fb._utils.classifier import Classifier
from captum.concept.fb._utils.data_iterator import get_concept_iterator_from_path
from captum.concept.fb._utils.common import concepts_to_str

from ....helpers.basic import BaseTest
from ....helpers.basic_models import BasicModel_ConvNet


def layer_attributions(model, layer_module, inputs, target, additional_forward_args):

    # attrib = LayerIntergratedGradients().attribute
    attrib, _, _ = captum_grad.compute_layer_gradients_and_eval(
        model, layer_module, inputs, target_ind=target
    )

    return attrib


def train_test_split(x_list, y_list, test_split=0.33):

    # Shuffle
    z_list = list(zip(x_list, y_list))

    # Split
    test_size = int(test_split * len(z_list))
    z_test, z_train = z_list[:test_size], z_list[test_size:]
    x_test, y_test = zip(*z_test)
    x_train, y_train = zip(*z_train)

    x_train = torch.stack(x_train)
    x_test = torch.stack(x_test)
    y_train = torch.stack(y_train)
    y_test = torch.stack(y_test)

    y_train[: len(y_train) // 2] = 0
    y_train[len(y_train) // 2 :] = 1

    y_test[: len(y_test) // 2] = 0
    y_test[len(y_test) // 2 :] = 1

    return x_train, x_test, y_train, y_test


class CustomClassifier(Classifier):
    r"""
        Linear Classifier class implementation for testing the Tests with
        Concept Activation Vectors (TCAVs), as in the paper:
            https://arxiv.org/pdf/1711.11279.pdf

        This class simulates the output of a Linear Classifier such as
        sklearn without actually using it.

    """

    def __init__(self):

        return

    def fit(self, inputs, labels):

        return

    def predict(self, inputs):

        # A tensor with dimensions n_inputs x (1 - test_split) x n_concepts
        # should be returned here.

        # Assemble a list with size inputs.shape[0], divided in 4 quarters
        # [0, 0, 0, ... | 1, 1, 1, ... | 0, 0, 0, ... | 1, 1, 1, ... ]
        pred = [0] * inputs.shape[0]

        # Store the shape of 1/4 of inputs.shape[0] (sh_4) and use it
        sh_4 = inputs.shape[0] / 4
        for i in range(1, 4, 2):

            from_ = round(i * sh_4)
            to_ = round((i + 1) * sh_4)

            pred[from_:to_] = [1] * (round((i + 1) * sh_4) - round(i * sh_4))

        pred = torch.tensor(pred)

        return pred

    def weights(self):

        return torch.tensor(
            [
                [
                    -0.2167,
                    -0.0809,
                    -0.1235,
                    -0.2450,
                    0.2954,
                    0.5409,
                    -0.2587,
                    -0.3428,
                    0.2486,
                    -0.0123,
                    0.2737,
                    0.4876,
                    -0.1133,
                    0.1616,
                    -0.2016,
                    -0.0413,
                ],
                [
                    -0.2167,
                    -0.0809,
                    -0.1235,
                    -0.2450,
                    0.2954,
                    0.5409,
                    -0.2587,
                    -0.3428,
                    0.2486,
                    -0.0123,
                    0.2737,
                    0.4876,
                    -0.1133,
                    0.1616,
                    -0.2016,
                    -0.0413,
                ],
            ],
            dtype=torch.float64,
        )


class CustomIterableDataset(IterableDataset):
    r"""
        Auxiliary class for iterating through an image dataset.
    """

    def __init__(self, get_tensor_from_filename_func, path):
        r"""
        Args:

            path (str): Path to dataset files
        """

        self.path = path
        self.file_itr = ["x"] * 100
        self.get_tensor_from_filename_func = get_tensor_from_filename_func

    def get_tensor_from_filename(self, filename):

        return self.get_tensor_from_filename_func(filename)

    def __iter__(self):

        mapped_itr = map(self.get_tensor_from_filename, self.file_itr)

        return mapped_itr


def get_tensor_from_filename(filename):

    file_tensor = torch.tensor(
        [
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    0.4556,
                    0.6323,
                ],
                [
                    0.3489,
                    0.4017,
                    0.0223,
                    0.1689,
                    0.2939,
                    0.5185,
                    0.6977,
                    0.8000,
                    0.1610,
                    0.2823,
                ],
                [
                    0.6816,
                    0.9152,
                    0.3971,
                    0.8742,
                    0.4194,
                    0.5529,
                    0.9527,
                    0.0362,
                    0.1852,
                    0.3734,
                ],
                [
                    0.3051,
                    0.9320,
                    0.1759,
                    0.2698,
                    0.1507,
                    0.0317,
                    0.2081,
                    0.9298,
                    0.7231,
                    0.7423,
                ],
                [
                    0.5263,
                    0.2437,
                    0.5846,
                    0.0332,
                    0.1387,
                    0.2422,
                    0.8155,
                    0.7932,
                    0.2783,
                    0.4820,
                ],
                [
                    0.8198,
                    0.9971,
                    0.6984,
                    0.5675,
                    0.8352,
                    0.2056,
                    0.5932,
                    0.1123,
                    0.1535,
                    0.2417,
                ],
                [
                    0.7262,
                    0.7011,
                    0.2038,
                    0.6511,
                    0.7745,
                    0.4369,
                    0.5191,
                    0.6159,
                    0.8102,
                    0.9801,
                ],
                [
                    0.1147,
                    0.3168,
                    0.6965,
                    0.9143,
                    0.9351,
                    0.9412,
                    0.5995,
                    0.0652,
                    0.5460,
                    0.1872,
                ],
                [
                    0.0340,
                    0.9442,
                    0.8802,
                    0.0012,
                    0.5936,
                    0.4158,
                    0.4177,
                    0.2711,
                    0.6923,
                    0.2038,
                ],
                [
                    0.6833,
                    0.7529,
                    0.8579,
                    0.6870,
                    0.0051,
                    0.1757,
                    0.7497,
                    0.6047,
                    0.1100,
                    0.2121,
                ],
            ]
        ]
    )

    return file_tensor


def get_inputs_tensor():

    input_tensor = torch.tensor(
        [
            [
                [
                    [
                        -1.1258e00,
                        -1.1524e00,
                        -2.5058e-01,
                        -4.3388e-01,
                        8.4871e-01,
                        6.9201e-01,
                        -3.1601e-01,
                        -2.1152e00,
                        3.2227e-01,
                        -1.2633e00,
                    ],
                    [
                        3.4998e-01,
                        3.0813e-01,
                        1.1984e-01,
                        1.2377e00,
                        1.1168e00,
                        -2.4728e-01,
                        -1.3527e00,
                        -1.6959e00,
                        5.6665e-01,
                        7.9351e-01,
                    ],
                    [
                        5.9884e-01,
                        -1.5551e00,
                        -3.4136e-01,
                        1.8530e00,
                        7.5019e-01,
                        -5.8550e-01,
                        -1.7340e-01,
                        1.8348e-01,
                        1.3894e00,
                        1.5863e00,
                    ],
                    [
                        9.4630e-01,
                        -8.4368e-01,
                        -6.1358e-01,
                        3.1593e-02,
                        -4.9268e-01,
                        2.4841e-01,
                        4.3970e-01,
                        1.1241e-01,
                        6.4079e-01,
                        4.4116e-01,
                    ],
                    [
                        -1.0231e-01,
                        7.9244e-01,
                        -2.8967e-01,
                        5.2507e-02,
                        5.2286e-01,
                        2.3022e00,
                        -1.4689e00,
                        -1.5867e00,
                        -6.7309e-01,
                        8.7283e-01,
                    ],
                    [
                        1.0554e00,
                        1.7784e-01,
                        -2.3034e-01,
                        -3.9175e-01,
                        5.4329e-01,
                        -3.9516e-01,
                        -4.4622e-01,
                        7.4402e-01,
                        1.5210e00,
                        3.4105e00,
                    ],
                    [
                        -1.5312e00,
                        -1.2341e00,
                        1.8197e00,
                        -5.5153e-01,
                        -5.6925e-01,
                        9.1997e-01,
                        1.1108e00,
                        1.2899e00,
                        -1.4782e00,
                        2.5672e00,
                    ],
                    [
                        -4.7312e-01,
                        3.3555e-01,
                        -1.6293e00,
                        -5.4974e-01,
                        -4.7983e-01,
                        -4.9968e-01,
                        -1.0670e00,
                        1.1149e00,
                        -1.4067e-01,
                        8.0575e-01,
                    ],
                    [
                        -9.3348e-02,
                        6.8705e-01,
                        -8.3832e-01,
                        8.9182e-04,
                        8.4189e-01,
                        -4.0003e-01,
                        1.0395e00,
                        3.5815e-01,
                        -2.4600e-01,
                        2.3025e00,
                    ],
                    [
                        -1.8817e00,
                        -4.9727e-02,
                        -1.0450e00,
                        -9.5650e-01,
                        3.3532e-02,
                        7.1009e-01,
                        1.6459e00,
                        -1.3602e00,
                        3.4457e-01,
                        5.1987e-01,
                    ],
                ]
            ],
            [
                [
                    [
                        -2.6133e00,
                        -1.6965e00,
                        -2.2824e-01,
                        2.7995e-01,
                        2.4693e-01,
                        7.6887e-02,
                        3.3801e-01,
                        4.5440e-01,
                        4.5694e-01,
                        -8.6537e-01,
                    ],
                    [
                        7.8131e-01,
                        -9.2679e-01,
                        -2.1883e-01,
                        -2.4351e00,
                        -7.2915e-02,
                        -3.3986e-02,
                        9.6252e-01,
                        3.4917e-01,
                        -9.2146e-01,
                        -5.6195e-02,
                    ],
                    [
                        -6.2270e-01,
                        -4.6372e-01,
                        1.9218e00,
                        -4.0255e-01,
                        1.2390e-01,
                        1.1648e00,
                        9.2337e-01,
                        1.3873e00,
                        -8.8338e-01,
                        -4.1891e-01,
                    ],
                    [
                        -8.0483e-01,
                        5.6561e-01,
                        6.1036e-01,
                        4.6688e-01,
                        1.9507e00,
                        -1.0631e00,
                        -7.7326e-02,
                        1.1640e-01,
                        -5.9399e-01,
                        -1.2439e00,
                    ],
                    [
                        -1.0209e-01,
                        -1.0335e00,
                        -3.1264e-01,
                        2.4579e-01,
                        -2.5964e-01,
                        1.1834e-01,
                        2.4396e-01,
                        1.1646e00,
                        2.8858e-01,
                        3.8660e-01,
                    ],
                    [
                        -2.0106e-01,
                        -1.1793e-01,
                        1.9220e-01,
                        -7.7216e-01,
                        -1.9003e00,
                        1.3068e-01,
                        -7.0429e-01,
                        3.1472e-01,
                        1.5739e-01,
                        3.8536e-01,
                    ],
                    [
                        9.6715e-01,
                        -9.9108e-01,
                        3.0161e-01,
                        -1.0732e-01,
                        9.9846e-01,
                        -4.9871e-01,
                        7.6111e-01,
                        6.1830e-01,
                        3.1405e-01,
                        2.1333e-01,
                    ],
                    [
                        -1.2005e-01,
                        3.6046e-01,
                        -3.1403e-01,
                        -1.0787e00,
                        2.4081e-01,
                        -1.3962e00,
                        -6.6144e-02,
                        -3.5836e-01,
                        -1.5616e00,
                        -3.5464e-01,
                    ],
                    [
                        1.0811e00,
                        1.3148e-01,
                        1.5735e00,
                        7.8143e-01,
                        -5.1107e-01,
                        -1.7137e00,
                        -5.1006e-01,
                        -4.7489e-01,
                        -6.3340e-01,
                        -1.4677e00,
                    ],
                    [
                        -8.7848e-01,
                        -2.0784e00,
                        -1.1005e00,
                        -7.2013e-01,
                        1.1931e-02,
                        3.3977e-01,
                        -2.6345e-01,
                        1.2805e00,
                        1.9395e-02,
                        -8.8080e-01,
                    ],
                ]
            ],
        ],
        requires_grad=True,
    )

    return input_tensor


def create_concept(concept_name, concept_id):

    concepts_path = "./dummy/concepts/" + concept_name + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
    concept_iter = get_concept_iterator_from_path(dataset)

    concept = Concept(id=concept_id, name=concept_name, data_iter=concept_iter)

    return concept


def create_concepts():

    # Function to create concept objects from a pre-set concept name list.

    concept_names = ["striped", "ceo", "random", "dotted"]

    concept_list = []
    concept_dict = defaultdict(Concept)

    for c, concept_name in enumerate(concept_names):
        concept = create_concept(concept_name, c)
        concept_list.append(concept)
        concept_dict[concept_name] = concept

    return concept_list, concept_dict


def create_TCAV(concepts, save_path):

    model = BasicModel_ConvNet()

    layers = ["conv2"]
    classifier = CustomClassifier()
    tcav = TCAV(
        model,
        concepts,
        layers,
        classifier,
        layer_attributions,
        train_test_split,
        save_path=save_path,
    )
    return tcav


def init_TCAV(save_path):

    # Create Concepts
    concepts, concepts_dict = create_concepts()

    # Create TCAV (with all but the last concept in "concepts")
    tcav = create_TCAV(concepts[:-1], save_path)

    # Generate Activations
    tcav.generate_all_activations()

    return tcav, concepts_dict


def remove_pkls(path):

    pkl_files = glob.glob(os.path.join(path, "*.pkl"))
    for pkl_file in pkl_files:
        os.remove(pkl_file)


class Test(BaseTest):
    r"""
    Class for testing the TCAV class through a sequence of operations:
    - Create the Concepts (random tensor generation simulation)
    - Create the TCAV class
    - Generate Activations
    - Compute the CAVs
    - Interpret (the images - simulated with random tensors)
    """

    # Init - Generate Activations
    def test_TCAV_1(self) -> None:

        # Create Concepts
        concepts, _ = create_concepts()
        for concept in concepts:

            self.assertEqual(len(concept.data_iter.dataset.file_itr), 100)

            total_batches = 0
            for data in concept.data_iter:
                total_batches += data.shape[0]
                self.assertEqual(data.shape[1:], torch.Size([1, 10, 10]))
            self.assertEqual(total_batches, 100)

        with tempfile.TemporaryDirectory() as tmpdirname:

            # Create TCAV
            tcav = create_TCAV(concepts, tmpdirname)

            # Generate Activations
            tcav.generate_all_activations()
            self.assertEqual(
                tcav.concept2av_map["striped"][tcav.layers[0]][0].shape,
                torch.Size([64, 16]),
            )
            self.assertEqual(
                tcav.concept2av_map["striped"][tcav.layers[0]][1].shape,
                torch.Size([36, 16]),
            )

    def compute_cavs_interpret(
        self,
        experimental_set_list,
        save_cavs,
        force_train,
        run_parallel,
        accs,
        sign_count,
        magnitude,
        remove_activation=False,
    ):

        with tempfile.TemporaryDirectory() as tmpdirname:

            tcav, concept_dict = init_TCAV(tmpdirname)

            experimental_set = []
            for concept_set in experimental_set_list:

                concepts = []
                for concept in concept_set:
                    self.assertTrue(concept in concept_dict)
                    concepts.append(concept_dict[concept])

                experimental_set.append(concepts)

            # Compute CAVs

            tcav.compute_cavs(
                experimental_set,
                save_cavs=save_cavs,
                force_train=force_train,
                run_parallel=run_parallel,
            )
            concepts_key = concepts_to_str(experimental_set[0])

            self.assertEqual(
                tcav.cavs[concepts_key][tcav.layers[0]].stats["weights"].shape,
                torch.Size([2, 16]),
            )

            self.assertAlmostEqual(
                tcav.cavs[concepts_key][tcav.layers[0]].stats["accs"],
                accs,
                delta=0.0001,
            )

            # Provoking a CAV absence by deleting the .pkl files and one
            # activation
            if remove_activation:
                remove_pkls(tmpdirname)
                tcav.concept2av_map["random"]["conv2"] = []

            # Interpret

            inputs = 100 * get_inputs_tensor()
            scores = tcav.interpret(
                inputs=inputs,
                experimental_set=experimental_set,
                target=0,
                run_parallel=run_parallel,
            )

            self.assertAlmostEqual(
                scores[concepts_key]["conv2"]["sign_count"][0], sign_count, delta=0.0001
            )

            self.assertAlmostEqual(
                scores[concepts_key]["conv2"]["magnitude"][0], magnitude, delta=0.0001
            )

    # Save CAVs, Force Train
    def test_TCAV_1_1_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            True,
            True,
            0.5000,
            0.5000,
            0.9512,
        )

    def test_TCAV_1_1_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"]], True, True, True, 0.5000, 0.5000, 0.9512
        )

    def test_TCAV_1_1_c(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"], ["striped", "ceo"]],
            True,
            True,
            True,
            0.5000,
            0.5000,
            0.9512,
        )

    # Non-existing concept in the experimental set ("dotted")
    def test_TCAV_1_1_d(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            True,
            True,
            0.5000,
            0.5000,
            0.9512,
        )

    # Do not save CAVs, Force Train
    def test_TCAV_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            True,
            True,
            0.5000,
            0.5000,
            0.9512,
        )

    # Do not save CAVs, Do not Force Train
    def test_TCAV_0_0(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            False,
            True,
            0.4848,
            0.5000,
            0.9512,
        )

    # Save CAVs, Do not Force Train
    def test_TCAV_1_0_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            False,
            True,
            0.4848,
            0.5000,
            0.9512,
        )

    # Non-existing concept in the experimental set ("dotted"), do Not Force Train
    def test_TCAV_1_0_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            False,
            True,
            0.4848,
            0.5000,
            0.9512,
        )

    # Save CAVs, Do not Force Train, Missing Activation
    def test_TCAV_1_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            False,
            True,
            0.4848,
            0.5000,
            0.9512,
            remove_activation=True,
        )

    ### Do not run parallel:

    # Save CAVs, Force Train
    def test_TCAV_x_1_1_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            True,
            False,
            0.5000,
            0.5000,
            0.9512,
        )

    def test_TCAV_x_1_1_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"]], True, True, False, 0.5000, 0.5000, 0.9512
        )

    def test_TCAV_x_1_1_c(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"], ["striped", "ceo"]],
            True,
            True,
            False,
            0.5000,
            0.5000,
            0.9512,
        )

    # Non-existing concept in the experimental set ("dotted")
    def test_TCAV_x_1_1_d(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            True,
            False,
            0.5000,
            0.5000,
            0.9512,
        )

    # Do not save CAVs, Force Train
    def test_TCAV_x_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            True,
            False,
            0.5000,
            0.5000,
            0.9512,
        )

    # Do not save CAVs, Do not Force Train
    def test_TCAV_x_0_0(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            False,
            False,
            0.4848,
            0.5000,
            0.9512,
        )

    # Save CAVs, Do not Force Train
    def test_TCAV_x_1_0_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            False,
            False,
            0.4848,
            0.5000,
            0.9512,
        )

    # Non-existing concept in the experimental set ("dotted"), do Not Force Train
    def test_TCAV_x_1_0_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            False,
            False,
            0.4848,
            0.5000,
            0.9512,
        )

    # Save CAVs, Do not Force Train, Missing Activation
    def test_TCAV_x_1_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            False,
            False,
            0.4848,
            0.5000,
            0.9512,
            remove_activation=True,
        )
