# std libs
from typing import TypeVar, List, Callable, Set
from random import shuffle
from math import floor

# other libs
from scipy.stats import beta

T = TypeVar('T')
Partition = List[Set[T]]


def random_partition(mentions: List[T], rng: Callable[[], float]) -> Partition:
    """
    Generates a random partitions of mentions.
    The size of the entities composing the partitions is randomly drawn following the random number
    generator rng.
    """
    shuffle(mentions)
    partitions = []
    while len(mentions) != 0:
        y = floor(rng() * len(mentions))+1
        partitions.append(set(mentions[:y]))
        mentions = mentions[y:]
    return partitions


def beta_partition(mentions: List[T], a: float, b: float) -> Partition:
    """
    Generates a random partitions of mentions, entity's sizes are randomly drawn following a
    beta distribution of parameter a, b.
    """
    return random_partition(mentions, beta(a, b).rvs)


def entity_partition(mentions: List[T]) -> Partition:
    """
    Return a Partition constitued of an unique entity containing each mention of mentions
    """
    if not mentions:
        return []
    return [{mention for mention in mentions}]


def singleton_partition(mentions: List[T]) -> Partition:
    """
    Return a Partition consitued of only singletons, one for each of mention of mentions
    """
    return [{mention} for mention in mentions]


def partition_to_sklearn_format(partition: Partition) -> List[int]:
    """
    Converts a Partition to the classification format used by sklearn
    examples:
    [{1}, {4, 3}, {2, 5, 6}} -> [0, 2, 1, 1, 2, 2]
    """
    tmp = {mention: n for n, entity in enumerate(partition) for mention in entity}
    return [v for k, v in sorted(tmp.items())]


def get_mentions(partition: Partition) -> List[T]:
    """
    Return the list of all mentions in a Partition
    """
    return [mention for entity in partition for mention in entity]
