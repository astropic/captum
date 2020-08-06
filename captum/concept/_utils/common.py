#!/usr/bin/env python3


def concepts_to_str(concepts):
    r"""
    Returns a string of concatenated (by "-") Concept names for the
    Concepts in concepts. Output example: "striped-random_0-random_1"

    Args:
        concepts (list[Concept]): a List of concept names to be
                    concatenated and used as a concepts key. These concept
                    names are respective to the Concept objects used for
                    the classifier train.
    Returns:
        A string of concatenated (by "-") Concept names.
        Ex.: "striped-random_0-random_1"
    """

    return "-".join([c.name for c in concepts])
