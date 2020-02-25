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

from utils import ScoreHolder, evaluates
from partition_utils import Partition, beta_partition, entity_partition, singleton_partition, get_mentions
from property_tests import symetry_test, singleton_test, entity_test, randomized_test, ancor_gold_randomized_test
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
) -> Iterator[ScoreHolder]:
    """
    Computes Scores for a partition against random partition generated with partition_generator
    """
    for n, k in enumerate(golds):
        syss: Iterator[Partition] = (partition_generator(get_mentions(k)) for _ in range(1))
        yield ScoreHolder.average(map(lambda r: evaluates(k, r), syss))


# start = time()
# #gen = scoress_average(score_partitions(K()))
# mentions = [i for i in range(100)]
# KK = (beta_partition(mentions, 1, 100) for _ in range(1))
# partition_generator = construct_partition  #partial(beta_partition, a = 1, b = 1)
# res = Scores.average(score_random_partitions(K(), partition_generator))
# print(time()-start)
# print(res)


# TODO all test in a similar format ?
def golds_vs_entity(golds: Iterator[Partition]) -> Tuple[ScoreHolder, ScoreHolder]:
    return ScoreHolder.avg_std(score_random_partitions(golds, entity_partition))


def golds_vs_singleton(golds: Iterator[Partition]) -> ScoreHolder:
    return ScoreHolder.average(score_random_partitions(golds, singleton_partition))


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


def scale_test(gold: Partition, sys: Partition) -> ScoreHolder:
    score_a = evaluates(gold, sys)
    score_b = evaluates(*duplicate_clusters(gold, sys))
    return score_a.compare(score_b)


def true_test(gold: Partition, sys: Partition) -> ScoreHolder:
    a = evaluates(gold, sys)
    t = a['true']
    return a.compare_t(t)









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




a = [{1, 2, 3, 4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13}, {14}]
z = [{1, 2, 3, 4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13, 14}]
y = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 14}, {10, 11, 12, 13}]
b = [{1, 2, 3, 4, 5, 9}, {6, 7, 8}, {10, 11, 12}, {14, 13}]
c = [{1, 2, 3, 4, 5, 14}, {6, 7, 8}, {10, 11, 12}, {9, 13}]
d = [{1, 2, 3, 4, 5, 13}, {6, 7, 14}, {10, 11, 12}, {8, 9}]
e = [{1, 2, 3, 4, 5, 14}, {6, 7, 13}, {10, 11, 12}, {8, 9}]
l = [b, c, d, e, z, y, a]
for i in l:
    print(evaluates(a, i))