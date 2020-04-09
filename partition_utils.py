# std libs
from typing import TypeVar, List, Callable, Set, Iterator
from random import shuffle, randint, random, choice
from math import floor, ceil, factorial, exp, isclose
from functools import reduce
from collections import defaultdict

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
        y = floor(rng() * len(mentions)) + 1
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


def all_partition_of_size(n: int) -> Iterator[Partition]:
    """
    Generates all partition of n mentions
    """
    if n == 1:
        yield [{1}]
    else:
        for partition in all_partition_of_size(n - 1):
            for e, part, in enumerate(partition + [set()]):
                yield partition[:e] + [part.union({n})] + partition[e + 1:]


def all_partition_up_to_size(n: int) -> Iterator[Partition]:
    """
    Generates all partition of n or less mentions.
    """
    def intern(n):
        if n == 1:
            yield 1, [{1}]
        else:
            for partition_size, partition in intern(n - 1):
                yield partition_size, partition
                if partition_size == n-1:
                    for e, part, in enumerate(partition + [set()]):
                        yield n, partition[:e] + [part.union({n})] + partition[e + 1:]
    return (partition for _, partition in intern(n))


def is_regular(partition: Partition) -> bool:
    """
    Returns True if the given partition is composed of only singletons and is not composed of only one entity
    """
    return (not contains_one_entity(partition)) and (not is_dual(partition))


def contains_singleton(partition: Partition):
    """
    Returns Ture if the given partition contains at least one singleton
    """
    return any((len(entity) == 1 for entity in partition))


def contains_one_entity(partition: Partition):
    """
    Returns True if the given  partition is composed of only one entity
    """
    return len(partition) == 1


def is_dual(partition: Partition):
    """
    Return True if the given partition is composed only of singletons
    """
    return all((len(entity) == 1 for entity in partition))


def introduce_randomness(partition: Partition):
    """
    randomize the entity of a random mention in the partition
    """
    res = [{*e} for e in partition]
    old_pos = randint(0, len(res)-1)
    part = res[old_pos]
    elem = choice([*part])
    res[old_pos] -= {elem}
    if len(part) == 0:
        res.remove(part)

    new_pos = randint(0, len(res))
    if new_pos == len(res):
        res.append({elem})
    else:
        res[new_pos] |= {elem}
    return res


def bell(n: int) -> int:
    """
    Returns the n'th Bell's number.
    This is the number of possible partitions of a set of n elements.
    """
    return ceil(sum((k ** n) / (factorial(k)) for k in range(2 * n)) / exp(1))


def r_part(mentions: List) -> Partition:
    """
    Generates a random partition of n elements
    """
    def q(n, u):
        return(bell(n) ** -1) * exp(-1) * (u ** n) / factorial(u)
    dic = defaultdict(list)
    n = len(mentions)
    firsts_q = [q(n, 1)]
    r = random()
    nb_urn = 1
    p = q(n, nb_urn)
    while r > p or isclose(r, p):
        nb_urn += 1
        p += q(n, nb_urn)
    for i in mentions:
        urn = random() // (1/nb_urn)
        dic[urn] += [i]
    return [{*entity} for entity in dic.values() if entity]
