# std libs
from typing import TypeVar, List, Callable, Set
from random import shuffle
from math import ceil

# other libs
from scipy.stats import beta


Partition = List[Set]


def random_partition(mentions: List, rng: Callable[[], float]) -> Partition:
    """
    Generates a random partitions of mentions.
    The size of the clusters composing the partitions is randomly drawn following the random number
    generator rng.
    """
    shuffle(mentions)
    partitions = []
    while len(mentions) != 0:
        y = ceil(rng() * len(mentions))
        partitions.append(set(mentions[:y]))
        mentions = mentions[y:]
    return partitions


def beta_partition(a: float, b: float, mentions: List,) -> Partition:
    """
    Generates a random partitions of mentions, which cluster sizes are randomly drawn following a
    beta distribution of parameter a, b.
    """
    return random_partition(mentions, beta(a, b).rvs)


def entity_partition(mentions: List) -> Partition:
    """
    Return the partition constitued of an unique entity
    """
    return [{m for m in mentions}]


def singleton_partition(mentions: List) -> Partition:
    """
    Return the partition consitued of only singletons
    """
    return [{m} for m in mentions]


def partition_to_sklearn_format(partition: Partition) -> List:
    """
    Converts a Partition to the classification format used by sklearn in a pure way
    examples:
    [{1}, {4, 3}, {2, 5, 6}} -> [0, 2, 1, 1, 2, 2]
    """
    tmp = {item: n for n, cluster in enumerate(partition) for item in cluster}
    return [v for k, v in sorted(tmp.items())]


def get_mentions(partition: Partition) -> List:
    """
    Return the list of all mentions in a Partition
    """
    return [mention for entity in partition for mention in entity]
