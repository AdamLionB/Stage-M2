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

#TODO (NOT USED) nice function, is there still a point to it tho ?
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
    if n == 0:
        yield []
    elif n == 1:
        yield [{1}]
    else:
        for partition in all_partition_of_size(n - 1):
            for e, part, in enumerate(partition + [set()]):
                yield partition[:e] + [part.union({n})] + partition[e + 1:]


#TODO (NOT USED) make it return things in the right order so it can be used ?
def all_partition_up_to_size(n: int) -> Iterator[Partition]:
    """
    Generates all partition of n or less mentions.
    """
    def intern(n):
        if n == 0:
            yield 0, []
        elif n == 1:
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


def contains_singleton(partition: Partition) -> bool:
    """
    Returns True if the given partition contains at least one singleton
    """
    return any((len(entity) == 1 for entity in partition))


def contains_one_entity(partition: Partition) -> bool:
    """
    Returns True if the given  partition is composed of only one entity
    """
    return len(partition) == 1


def is_dual(partition: Partition) -> bool:
    """
    Returns True if the given partition is composed only of singletons
    """
    return all((len(entity) == 1 for entity in partition))


# TODO (NOT USED) remove ?
def get_partition_size(partition: Partition) -> int:
    """
    Returns the number of mentions in the partition
    """
    return reduce(lambda x, y: x + len(y), partition, 0)


def introduce_randomness(partition: Partition):
    """
    randomize the entity of a random mention in the partition
    """
    elem = choice(get_mentions(partition))
    res = [{*e} - {elem} for e in partition if {*e} != {elem}]

    new_pos = randint(0, len(res))
    if new_pos == len(res):
        res.append({elem})
    else:
        res[new_pos] |= {elem}
    return res

# FIXME floats <3
def bell(n: int) -> int:
    """
    Returns the n'th Bell's number.
    This is the number of possible partitions of a set of n elements.

    Sometimes this function will return bell's number +1,
    this is due to floating point error.  In this application this is not an issue
    since the only use of this function adds even more floating point issues and
    such a precision is not needed anyway.

    https://en.wikipedia.org/wiki/Dobiński%27s_formula
    """
    return ceil(sum((k ** n) / (factorial(k)) for k in range(2 * n)) / exp(1))


def r_part(mentions: List) -> Partition:
    """
    Generates a truly random partition from the mention list
    By truly random we mean that this function follows a uniform distribution,
    for a given number of mention each partition is as likely as any other to be generated

    Stam, A. J. (1983). Generation of a random partition of a finite set by an urn model.
    """
    def q(n, u):
        return (bell(n) ** -1) * exp(-1) * (u ** n) / factorial(u)
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
