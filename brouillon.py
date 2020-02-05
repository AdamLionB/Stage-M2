from __future__ import annotations
import sys as sys_lib
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import beta
from random import shuffle, random, randint
from math import ceil
import pathlib
import os
from time import time
from copy import copy
from typing import List, Set, Callable, Iterator, Tuple, Union, Optional, Dict, Any, Generic, TypeVar
from itertools import zip_longest, cycle, combinations, chain, islice
from functools import reduce, partial, singledispatchmethod
from collections import defaultdict
from enum import Enum, auto
from inspect import signature
import math

sys_lib.path.insert(1, '../scorch')
from scorch.main import METRICS, clusters_from_json
import scorch.scores as scores
from scorch.scores import links_from_clusters


def kaapa(k: Partition, r: Partition):
    C_k, N_k = links_from_clusters(k)
    C_r, N_r = links_from_clusters(r)

    tp_c = len(C_k.intersection(C_r))
    tp_n = len(N_k.intersection(N_r))
    c_k, n_k = len(C_k), len(N_k)
    c_r, n_r = len(C_r), len(N_r)

    return ((tp_c + tp_n) / (c_k + n_k) - (((c_k * c_r) + (n_k * n_r)) / (c_k + n_k) ** 2)) / (
            1 - (((c_k * c_r) + (n_k * n_r)) / (c_k + n_k) ** 2))


def true_r(gold: Partition, sys: Partition) -> float:
    n = 0
    for c_r in sys:
        if c_r in gold:
            n += 1
    return n / len(sys)


def true_p(gold: Partition, sys: Partition) -> float:
    n = 0
    for c_r in gold:
        if c_r in sys:
            n += 1
    return n / len(gold)


def true_f1(gold: Partition, sys: Partition) -> Tuple[float, float, float]:
    """
    Computes the true f1 for the (gold, sys) the true f1 is computed based on the set correctly resolved
    """
    r = true_r(gold, sys)
    p = true_p(gold, sys)
    return r, p, 2 * r * p / (r + p) if r + p != 0 else 0


def weighted_r(gold: Partition, sys: Partition) -> float:
    n = 0
    nb_mentions = 0
    for c_r in sys:
        if c_r in gold:
            n += len(c_r)
        nb_mentions += len(c_r)
    return n / nb_mentions


def weighted_p(gold: Partition, sys: Partition) -> float:
    n = 0
    nb_mentions = 0
    for c_r in gold:
        if c_r in sys:
            n += len(c_r)
        nb_mentions += len(c_r)
    return n / nb_mentions


def weighted_f1(gold: Partition, sys: Partition) -> Tuple[float, float, float]:
    """
    Computes the weighted f1 for the (gold, sys) the true f1 is
    computed based on the set correctly resolved weighted by their size
    """
    r = weighted_r(gold, sys)
    p = weighted_p(gold, sys)
    return r, p, 2 * r * p / (r + p) if r + p != 0 else 0


# print(tp_c, tp_n, c_k, n_k, c_r, n_r)

T = TypeVar('T')
Partition = List[Set[T]]
"""
Alias for a list of sets all composed of a same T type object.
"""


# TODO vérifier que ça marche
# TODO opti si nécessaire
def construct_partition(mentions: List, p=0.01) -> Partition:
    """
    Construct a random partition by giving a probability p for each possible link to be created
    """
    shuffle(mentions)
    partitions: Dict[Any, Set] = {}
    tmp = defaultdict(set)
    heads = {}
    for m in mentions:
        heads[m] = m
    for a, b in combinations(mentions, 2):
        if random() < p:
            heads[b] = heads[a]
            tmp[a].add(b)
    for m in mentions:
        partitions[heads[m]] = partitions.setdefault(heads[m], set()).union(tmp[m])
    partitions: Partition = [v.union([k]) for k, v in partitions.items()]
    return partitions


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


# TODO beta distribution vraiment idéal ?
def beta_partition(mentions: List, a: float, b: float) -> Partition:
    """
    Generates a random partitions of mentions, which cluster sizes are randomly drawn following a
    beta distribution of parameter a, b.
    """
    return random_partition(mentions, beta(a, b).rvs)


def entity_partition(mentions: List) -> Partition:
    """
    Return the partition constitued of an unique entity
    """
    return [{m for m in mentions}]  # random_partition(mentions, lambda: 1)


def singleton_partition(mentions: List) -> Partition:
    """
    Return the partition consitued of only singletons
    """
    return [{m} for m in mentions]  # random_partition(mentions, lambda: 1 / (len(mentions) + 1))


class SuchRandom:
    def __init__(self: SuchRandom):
        self.seed = cycle([1, 2, 3, 4])

    def __call__(self: SuchRandom):
        return 1 / next(self.seed)


such_random = SuchRandom()


def wow_partition(mentions: List) -> Partition:
    cluster = []
    while len(mentions) != 0:
        y = ceil(such_random() * len(mentions))
        cluster.append(set(mentions[:y]))
        mentions = mentions[y:]
    return cluster


def get_mentions(partition: Partition) -> List[T]:
    """
    Return the list of all mentions in a Partition
    """
    return [mention for entity in partition for mention in entity]


SK_METRICS = {
    'ARI': metrics.adjusted_rand_score,
    'HCV': metrics.homogeneity_completeness_v_measure,
    'AMI': metrics.adjusted_mutual_info_score,
    'FM': metrics.fowlkes_mallows_score
}


class Scores:
    """
    Dictionnary linking a str as key to a tuple of values
    """

    def __init__(self, dic: Dict[str, Tuple[float, ...]]):
        self.dic = dic

    def __getitem__(self, item: str) -> Tuple[float, ...]:
        return self.dic[item]

    def __add__(self, other: Scores) -> Scores:
        """
        Adds two Scores together and output resulting score in a pure way.
        """
        return Scores({
            sk: tuple((x + y for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def __sub__(self, other: Scores) -> Scores:
        """
        Substract the other Scores to self and output resulting Scores in a pure way.
        """
        return Scores({
            sk: tuple((x - y for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def __mul__(self, scalar: float) -> Scores:
        """
        Multiply a Scores by a scalar and output the resulting Scores in a pure way.
        """
        return Scores({
            k: tuple((x * scalar for x in v))
            for k, v in self.dic.items()
        })

    def __truediv__(self, scalar: float) -> Scores:
        """
        Divides a Scores by a scalar and output the resulting Scores in a pure way.
        """
        return Scores({
            k: tuple((x / scalar for x in v))
            for k, v in self.dic.items()
        })

    def __pow__(self, power: float, modulo=None):
        """
        Raise a Scores to the given power and output the resulting Scores in a pure way.
        """
        return Scores({
            k: tuple((x ** power for x in v))
            for k, v in self.dic.items()
        })

    def __str__(self) -> str:
        res = '\t\tF\tP\tR'
        for k, v in self.dic.items():
            res += f'\n{k}\t:'
            for e in reversed(v):
                if issubclass(type(e),float):
                    res += f'\t{e:.2f}'
                else:
                    res += f'\t{e}'
        return res

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def average(scoress: Iterator[Scores]) -> Scores:
        """
        Consume a Scores' iterator and output its average Scores.
        """
        res = next(scoress)
        count = 1
        for scores in scoress:
            res += scores
            count += 1
        return res / count

    @staticmethod
    def avg_std(scoress: Iterator[Scores]) -> Tuple[Scores, Scores]:
        """
        Consume a Scores' iterator and output its average and standard deviation Scores.
        """
        regular_sum = next(scoress)
        squared_sum = regular_sum ** 2
        count = 1
        for scores in scoress:
            regular_sum += scores
            squared_sum += scores ** 2
            count += 1
        return (regular_sum / count,
                ((squared_sum * count - regular_sum ** 2) / (count * (count - 1))) ** (1 / 2))

    def compare(self, other: Scores) -> Scores:
        return Scores({
            sk: tuple((Growth.compare(x, y) for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def compare_t(self, t: Tuple) -> Scores:
        return Scores({
            k: reversed(tuple((Growth.compare(x, y) for x, y in zip(v[::-1], t[::-1]))))
            for k, v in self.dic.items()
        })

    @staticmethod
    def growth(scoress: Iterator[Scores]) -> Scores:
        res = next(scoress)
        for scores in scoress:
            res += scores
        return res


class Growth(Enum):
    STRICT_INCR = auto()
    INCR = auto()
    CONST = auto()
    DECR = auto()
    STRICT_DECR = auto()
    NON_MONOTONIC = auto()

    def __add__(self, other: Growth) -> Growth:
        """
        ughh...
        """
        if self is other:
            return self
        if self is Growth.NON_MONOTONIC or other is Growth.NON_MONOTONIC:
            return Growth.NON_MONOTONIC
        elif self is Growth.INCR or self is Growth.STRICT_INCR:
            if other is Growth.DECR or other is Growth.STRICT_DECR:
                return Growth.NON_MONOTONIC
            else:
                return Growth.INCR
        elif self is Growth.DECR or self is Growth.STRICT_DECR:
            if other is Growth.INCR or other is Growth.STRICT_INCR:
                return Growth.NON_MONOTONIC
            else:
                return Growth.DECR
        if other is Growth.DECR or other is Growth.STRICT_DECR:
            return Growth.DECR
        else:
            return Growth.INCR

    def __truediv__(self, other: Any):
        return self

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self is Growth.CONST:
            return '='
        if self is Growth.NON_MONOTONIC:
            return '~'
        if self is Growth.INCR:
            return '+'
        if self is Growth.STRICT_INCR:
            return '++'
        if self is Growth.DECR:
            return '-'
        if self is Growth.STRICT_DECR:
            return '--'


    @staticmethod
    def compare(a: float, b: float):
        if a > b:
            return Growth.STRICT_DECR
        if a < b:
            return Growth.STRICT_INCR
        return Growth.CONST


def to_tuple(e: Union[Any, Tuple[Any]]) -> Tuple[Any]:
    """
    Output the input as tuple in a pure way.
    If the input was a tuple, return it. If it wasn't, puts it in a tuple then return it.
    """
    if type(e) == tuple:
        return e
    return e,


def partition_to_sklearn_format(partition: Partition) -> List:
    """
    Converts a Partition to the classification format used by sklearn in a pure way
    examples:
    [{1}, {4, 3}, {2, 5, 6}} -> [0, 2, 1, 1, 2, 2]
    """
    tmp = {item: n for n, cluster in enumerate(partition) for item in cluster}
    return [v for k, v in sorted(tmp.items())]


# TODO purify
def evaluate(gold: Partition, sys: Partition) -> Scores:
    """
    Computes metrics scores for a (gold, sys) and outputs it as a Scores
    """
    res = {}
    for name, metric in METRICS.items():
        res[name] = to_tuple(metric(gold, sys))
    res['conll'] = to_tuple(scores.conll2012(gold, sys))
    res['true'] = to_tuple(true_f1(gold, sys))
    res['w'] = to_tuple(weighted_f1(gold, sys))
    gold = partition_to_sklearn_format(gold)
    sys = partition_to_sklearn_format(sys)
    if len(gold) == len(sys):
        for name, metric in SK_METRICS.items():
            res[name] = to_tuple(metric(gold, sys))
    return Scores(res)


# TODO generalise for any json format
# TODO generalise for any format ? (with a given read method)
def iter_ancor() -> Iterator[Partition]:
    """
    Iterates over the ancor corpus
    """
    for n, file in enumerate(os.listdir('../ancor/json/')):
        # if n > 0 : break
        yield clusters_from_json(open('../ancor/json/' + file))


# TODO ça marche ?
def introduce_randomness(partition: Partition):
    """
    randomize the entity of a random mention in the partition
    """
    size = reduce(lambda x, y: x + len(y), partition, 0)
    old_pos = randint(0, size - 1)
    new_pos = randint(0, size - 1)
    print(size, old_pos, new_pos)
    cursor = 0
    elem: Any = None
    for cluster in partition:
        if cursor <= old_pos < cursor + len(cluster):
            for c, e in enumerate(cluster, cursor):
                if old_pos == c:
                    elem = e
                    break
            cluster.remove(elem)
            break
        else:
            cursor += len(cluster)
    print(elem)
    cursor = 0
    for cluster in partition:
        if cursor <= new_pos < cursor + len(cluster):
            cluster.add(elem)
            break
        else:
            cursor += len(cluster)
    return partition


def score_random_partitions(
        golds: Iterator[Partition],
        partition_generator: Callable[[List], Partition]
) -> Iterator[Scores]:
    """
    Computes Scores for a partition against random partition generated with partition_generator
    """
    for n, k in enumerate(golds):
        syss: Iterator[Partition] = (partition_generator(get_mentions(k)) for _ in range(1))
        yield Scores.average(map(lambda r: evaluate(k, r), syss))


# start = time()
# #gen = scoress_average(score_partitions(K()))
# mentions = [i for i in range(100)]
# KK = (beta_partition(mentions, 1, 100) for _ in range(1))
# partition_generator = construct_partition  #partial(beta_partition, a = 1, b = 1)
# res = Scores.average(score_random_partitions(K(), partition_generator))
# print(time()-start)
# print(res)


# TODO all test in a similar format ?
def golds_vs_entity(golds: Iterator[Partition]) -> Tuple[Scores, Scores]:
    return Scores.avg_std(score_random_partitions(golds, entity_partition))


def golds_vs_singleton(golds: Iterator[Partition]) -> Scores:
    return Scores.average(score_random_partitions(golds, singleton_partition))


# TODO comment that
def duplicate_clusters(gold: Partition, sys: Partition) -> Tuple[Partition, Partition]:
    mentions = list(set().union(get_mentions(gold), get_mentions(sys)))
    new_mentions = {}
    cast = type(mentions[0])
    for m in mentions:
        while (r := cast(randint(-1e6, 1e6))) in mentions or r in new_mentions:
            pass
        new_mentions[m] = r
    it = chain(iter(mentions), (new_mentions[m] for m in mentions))
    new_gold = [{next(it) for _ in c} for c in chain(gold, gold)]
    it = chain(iter(mentions), (new_mentions[m] for m in mentions))
    new_sys = [{next(it) for _ in c} for c in chain(sys, sys)]
    return new_gold, new_sys


def scale_test(gold: Partition, sys: Partition) -> Scores:
    score_a = evaluate(gold, sys)
    score_b = evaluate(*duplicate_clusters(gold, sys))
    return score_a.compare(score_b)


def symetry_test(gold: Partition, sys: Partition) -> Scores:
    scores_a = evaluate(gold, sys)
    scores_b = evaluate(sys, gold)
    return scores_a.compare(scores_b)


def singleton_test(gold: Partition) -> Scores:
    return evaluate(gold, singleton_partition(get_mentions(gold)))


def entity_test(gold: Partition) -> Scores:
    return evaluate(gold, entity_partition(get_mentions(gold)))


def identity_test(gold: Partition) -> Scores:
    return evaluate(gold, gold)


def triangle_test(a: Partition, b: Partition, c: Partition) -> Scores:
    return Scores.compare(evaluate(a, c), evaluate(a, b) + evaluate(b, c))


def true_test(gold: Partition, sys: Partition) -> Scores:
    a = evaluate(gold, sys)
    t = a['true']
    return a.compare_t(t)


def r_test(
        test: Callable[[Partition, ...], Scores],
        partition_generators: Tuple[Callable[[list], Partition], ...] = None,
        std: bool = False
) -> Union[Tuple[Scores, Scores], Scores]:
    def intern(m) -> Iterator[Scores]:
        n = len(signature(test).parameters)
        for _ in range(10):
            if partition_generators is None:
                a = (beta_partition(m, 1, 1) for _ in range(n))
            else:
                if len(partition_generators) != n:
                    # TODO exception
                    raise Exception()
                a = (part() for part in partition_generators)
            yield test(*a)
    mentions = list(range(100))
    if std:
        return Scores.avg_std(intern(mentions))
    else:
        return Scores.average(intern(mentions))


def fixed_gold_test(
        test: Callable[[Partition, ...], Scores],
        it: Iterator[Partition],
        partition_generators: Tuple[Callable[[list], Partition], ...] = None,
        std: bool = False
) -> Union[Tuple[Scores, Scores], Scores]:
    def intern() -> Iterator[Scores]:
        n = len(signature(test).parameters)-1
        for gold in it:
            m = get_mentions(gold)
            if partition_generators is None:
                a = (beta_partition(m, 1, 1) for _ in range(n))
            else:
                if len(partition_generators) != n:
                    # TODO exception
                    raise Exception()
                a = (part() for part in partition_generators)
            yield test(gold, *a)
    if std:
        return Scores.avg_std(intern())
    else:
        return Scores.average(intern())


def ancor_test(
        test: Callable[[Partition], Scores],
        std: bool = False
) -> Union[Tuple[Scores, Scores], Scores]:
    return fixed_gold_test(test, iter_ancor(), std)

# exemple blanc
# k = [{1}, {2}, {3}, {4}, {5, 12, 14}, {6}, {7, 9}, {8}, {10}, {11}, {13}]
# r = [{1}, {2}, {3}, {4, 6}, {5, 12}, {7, 9, 14}, {8}, {10}, {11}, {13}]
# r2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}]
# r3 = [{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}]
#
# R = [r1]  # [r1, r2, r3]


# exemple lea
# k = [{'a', 'b', 'c'}, {'d', 'e', 'f', 'g'}]
# r = [{'a', 'b'}, {'c', 'd'},{'f', 'g', 'h', 'i'}]


# exemple à la mano
# k = [{1}, {2, 3}, {4, 5, 6}, {13, 14}]
# r = [{1}, {3}, {2, 4, 5, 6}, {13, 14}]


# print(scale_test, '\n', r_test(scale_test))
# print(symetry_test, '\n', r_test(symetry_test))
# print(singleton_test, '\n', ancor_test(singleton_test, std=True))
# print(entity_test, '\n', ancor_test(entity_test, std=True))
# print(singleton_test, '\n', r_test(singleton_test, std=False))
# print(entity_test, '\n', r_test(entity_test, std=False))
# print(triangle_test, '\n', r_test(triangle_test))
# print(identity_test, '\n', r_test(identity_test))
# print(true_test, '\n',  r_test(true_test))
# print(reduce(lambda x, y: x+y, (len(beta_partition(list(range(100)), 1, 1)) for _ in range(100)))/100)

# dic = {}
# s = 0
# for partition in iter_ancor():
#     for cluster in partition:
#         s+= len(cluster)
#          #dic.setdefault(len(cluster), 0)
#          #dic[len(cluster)] +=1
#
# print(s/455)
# for k, v in sorted(dic.items()):
#     print(k,v)

# print(reduce(lambda x,y : x+y,(len(construct_partition(list(range(100)),1/28)) for _ in range(100)))/100)
# print(x)
# print(len(x))
# print(construct_partition(list(range(100)),1/28))


# for gold in iter_ancor():
#     rc_wn = 0
#     n_mentions = 0
#     for entity in gold:
#         rc_wn += (len(entity) * (len(entity) - 1)) / 2
#         n_mentions += len(entity)
#     L = (n_mentions * (n_mentions - 1)) / 2
#     wc_rn = L - rc_wn
#     print(L, rc_wn, wc_rn)
#     e = (rc_wn + wc_rn) / L
#     print(METRICS['BLANC'](gold, beta_partition(get_mentions(gold), 1, 1)))
#     break