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
from typing import List, Set, Callable, Iterator, Tuple, Union, Optional, Dict, Any
from itertools import zip_longest, cycle, combinations
from functools import reduce, partial, singledispatchmethod
from collections import defaultdict

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


# print(tp_c, tp_n, c_k, n_k, c_r, n_r)

Partition = List[Set]


def construct_partition(mentions: List, p=0.01) -> Partition:
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


def random_partition(mentions: List, distrib: Callable[[], float]) -> Partition:
    """
    Generates a random partitions of mentions.
    The size of the clusters composing the partitions is randomly drawn following the random
    generator distrib.
    """
    shuffle(mentions)
    partitions = []
    while len(mentions) != 0:
        y = ceil(distrib() * len(mentions))
        partitions.append(set(mentions[:y]))
        mentions = mentions[y:]
    return partitions


def beta_partition(mentions: List, a: float, b: float) -> Partition:
    """
    Generates a random partitions of mentions, which cluster sizes are randomly drawn following a
    beta distribution of parameter a, b.
    """
    return random_partition(mentions, beta(a, b).rvs)


def entity_partition(mentions: List) -> Partition:
    return random_partition(mentions, lambda: 1)


def singleton_partition(mentions: List) -> Partition:
    return random_partition(mentions, lambda: 1 / (len(mentions) + 1))


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


def get_mentions(partition: Partition) -> List:
    return [mention for entity in partition for mention in entity]


SK_METRICS = {
    'ARI': metrics.adjusted_rand_score,
    'HCV': metrics.homogeneity_completeness_v_measure,
    'AMI': metrics.adjusted_mutual_info_score,
    'FM': metrics.fowlkes_mallows_score
}


# METRICS['ARI'] = metrics.adjusted_rand_score


# k = [{1}, {2}, {3}, {4}, {5, 12, 14}, {6}, {7, 9}, {8}, {10}, {11}, {13}]
# r1 = [{1}, {2}, {3}, {4, 6}, {5, 12}, {7, 9, 14}, {8}, {10}, {11}, {13}]
# r2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}]
# r3 = [{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}]
#
# R = [r1]  # [r1, r2, r3]


class Scores:
    """
    Dictionnary linking a str as key to a tuple of values
    """
    def __init__(self, dic: Dict[str, Tuple[float, ...]]):
        self.dic = dic

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
        res = ''
        for k, v in self.dic.items():
            res += f'{k}\t: {v}\n'
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
            res = res + scores
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
                ((squared_sum * count - regular_sum ** 2)/ (count * (count - 1))) ** (1 / 2))


def to_tuple(e: Union[Any, Tuple[Any]]) -> Tuple[Any]:
    """
    Output the input as tuple in a pure way.
    If the input was a tuple, return it. If it wasn't, puts it in a tuple then return it.
    """
    if type(e) == tuple:
        return e
    return e,


def partition_to_classif(partition: Partition) -> List:
    tmp = {item: n for n, cluster in enumerate(partition) for item in cluster}
    return [v for k, v in sorted(tmp.items())]


def get_scores(gold: Partition, sys: Partition) -> Scores:
    res = {}
    for name, metric in METRICS.items():
        res[name] = to_tuple(metric(gold, sys))
    res['conll'] = to_tuple(scores.conll2012(gold, sys))
    gold = partition_to_classif(gold)
    sys = partition_to_classif(sys)
    for name, metric in SK_METRICS.items():
        res[name] = to_tuple(metric(gold, sys))
    return Scores(res)


def iter_ancor() -> Iterator[Partition]:
    for n, file in enumerate(os.listdir('../ancor/json/')):
        # if n > 0 : break
        yield clusters_from_json(open('../ancor/json/' + file))


def introduce_randomness(partition: Partition):
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
        parition_generator: Callable[[List], Partition]
) -> Iterator[Scores]:
    for n, k in enumerate(golds):
        syss: Iterator[Partition] = (parition_generator(get_mentions(k)) for _ in range(1))
        yield Scores.average(map(lambda r: get_scores(k, r), syss))


# start = time()
# #gen = scoress_average(score_partitions(K()))
# mentions = [i for i in range(100)]
# KK = (beta_partition(mentions, 1, 100) for _ in range(1))
# partition_generator = construct_partition  #partial(beta_partition, a = 1, b = 1)
# res = Scores.average(score_random_partitions(K(), partition_generator))
# print(time()-start)
# print(res)

def golds_vs_entity(golds: Iterator[Partition]) -> Tuple[Scores, Scores]:
    return Scores.avg_std(score_random_partitions(golds, entity_partition))


def golds_vs_singleton(golds: Iterator[Partition]) -> Scores:
    return Scores.average(score_random_partitions(golds, singleton_partition))



print(golds_vs_entity(iter_ancor()))
