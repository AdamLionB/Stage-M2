from __future__ import annotations

# std lib
from typing import Callable, Tuple, Union, Iterator, Optional
from inspect import signature
from os import listdir
from itertools import product
from math import isclose

# scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions,\
    all_partition_of_size
from utils import ScoreHolder, evaluates, BinaryResult


def a1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.has_passed_test(not isclose(x, y) and x > y)
    gold = [{1, 2}, {3}, {4}]
    sys = [{1, 2}, {3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.has_passed_test(not isclose(x, y) and x > y)
    gold = [{3}, {4}]
    sys = [{3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a3_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.has_passed_test(not isclose(x, y) and x > y)
    gold = [{1, 2}, {3}, {4}]
    sys = [{1, 2, 3}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def b1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(not isclose(x, y) and x < y)
    gold = [{1, 2}, {3, 4, 5}]
    sys1 = [{1, 2, 3}, {4, 5}]
    sys2 = [{1, 2, 3, 4, 5}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def b2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(not isclose(x, y) and x < y)
    gold = [{1, 2, 3, 4, 5}, {6, 7}]
    sys1 = [{1, 2, 3, 4}, {5, 6, 7}]
    sys2 = [{1, 2}, {3, 4, 5}, {6, 7}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def d1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(isclose(x, y))
    gold1 = [{1, 2}, {3, 4}, {5, 6}]
    sys1 = [{1, 3}, {2, 4}, {5, 6}]
    gold2 = [{1, 2}, {3, 4}, {5, 6, 7, 8}]
    sys2 = [{1, 3}, {2, 4}, {5, 6, 7, 8}]
    return ScoreHolder.apply(evaluates(gold1, sys1), intern, evaluates(gold2, sys2))


def d2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(isclose(x, y))
    gold = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys1 = [{1, 2, 3, 4, 6}, {5, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys2 = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 14}, {13, 15, 16}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def f_test(gold: Partition, sys:Partition) -> ScoreHolder:
    def intern(x: float) -> BinaryResult:
        return BinaryResult.get_binary_result(not isclose(x, 0))
    return ScoreHolder.apply_to_values(evaluates(gold, sys), intern)


def symetry_test(gold: Partition, sys: Partition) -> ScoreHolder:
    """
    evaluates whether score(gold, sys) = score(sys, gold)
    """
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.has_passed_test(isclose(x, y))
    return ScoreHolder.apply(evaluates(gold, sys), intern, evaluates(sys, gold))


def singleton_test(gold: Partition) -> ScoreHolder:
    """
    evaluates the score of a partition against a partition composed only of singletons
    """
    return evaluates(gold, singleton_partition(get_mentions(gold)))


def entity_test(gold: Partition) -> ScoreHolder:
    """
    evaluates the score of a partition against a partition composed of only one entity
    """
    return evaluates(gold, entity_partition(get_mentions(gold)))


def non_identity_test(gold: Partition, sys: Partition) -> Optional[ScoreHolder]:
    """
    evaluates whether score(gold, sys) != 1 with gold != sys
    """
    def intern(x: float):
        return BinaryResult.has_passed_test(not isclose(x, 1))
    if gold == sys:
        return None
    return evaluates(gold, sys).apply_to_values(intern)


def identity_test(gold: Partition) -> ScoreHolder:
    """
    evaluates whether score(gold, gold) = 1
    """
    def intern(x: float):
        return BinaryResult.has_passed_test(isclose(x, 1))
    return evaluates(gold, gold).apply_to_values(intern)


def triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder:
    """
    evaluates whether the (similarity) triangle inequality is respected
    s(a,b) + s(b,c) <= s(b,b) + s(a,b)
    """
    def intern(x: float, y:float) -> BinaryResult:
        return BinaryResult.has_passed_test(isclose(x, y) or x < y)
    return ScoreHolder.apply(evaluates(a, b) + evaluates(b, c), intern, evaluates(b, b) + evaluates(a, c))


def randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    """
    Apply the given test function to partitions randomly generated with partition_generators
    """
    def intern(m) -> Iterator[ScoreHolder]:
        n_args = len(signature(test_func).parameters)
        # repeats the test 'repetitions' times
        for _ in range(repetitions):
            # if no partition_generator is given partition will follow a beta partition
            if partition_generators is None:
                a = (beta_partition(*beta_param, m) for _ in range(n_args))
            else:
                if len(partition_generators) != n_args:
                    raise Exception(f'got {len(partition_generators)} partition generator, expected {n_args}')
                a = (part(m) for part in partition_generators)
            res = test_func(*a)
            if res is not None:
                yield res
    mentions = list(range(100))
    if std:
        return ScoreHolder.avg_std(intern(mentions))
    else:
        return ScoreHolder.average(intern(mentions))


def fixed_gold_randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder]],
        it: Iterator[Partition],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    def intern() -> Iterator[ScoreHolder]:
        n_agrs = len(signature(test_func).parameters) - 1
        for gold in it:
            m = get_mentions(gold)
            # repeats the test 'repetitions' times
            for _ in range(repetitions):
                # if no partition_generator is given partition will follow a beta partition
                if partition_generators is None:
                    a = (beta_partition(*beta_param, m) for _ in range(n_agrs))
                else:
                    if len(partition_generators) != n_agrs:
                        raise Exception(f'got {len(partition_generators)} partition generator, expected {n_agrs}')
                    a = (part(m) for part in partition_generators)
            res = test_func(gold, *a)
            if res is not None:
                yield res
    if std:
        return ScoreHolder.avg_std(intern())
    else:
        return ScoreHolder.average(intern())


def ancor_gold_randomized_test(
        test_func: Callable[[Partition], Optional[ScoreHolder]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    return fixed_gold_randomized_test(test_func, iter_ancor(), partition_generators=partition_generators, std=std, repetitions=repetitions, beta_param=beta_param)


def all_partitions_test(
        test_func: Callable[[Partition], Optional[ScoreHolder]],
        std: bool = False,
        start: int = 1,
        end: int = 2
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    def intern(i) -> Iterator[ScoreHolder]:
        n_args = len(signature(test_func).parameters)
        for args in product(all_partition_of_size(i), repeat=n_args):
            res = test_func(*args)
            if res is not None:
                yield res
    if std:
        return ScoreHolder.avg_std(ScoreHolder.average(intern(i)) for i in range(start, end))
    else:
        return  ScoreHolder.average(ScoreHolder.average(intern(i))for i in range(start, end))



# TODO generalise for any json format
# TODO generalise for any format ? (with a given read method)
def iter_ancor() -> Iterator[Partition]:
    """
    Iterates over the ancor corpus
    """
    for n, file in enumerate(listdir('../ancor/json/')):
        # if n > 0 : break
        yield clusters_from_json(open('../ancor/json/' + file))
