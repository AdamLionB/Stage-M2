from __future__ import annotations
# std lib
from typing import Callable, Tuple, Union, Iterator, Optional
from inspect import signature
from os import listdir
from itertools import product
from math import isclose

#scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions,\
    all_partition_of_size
from utils import ScoreHolder, evaluate, BinaryResult


def symetry_test(gold: Partition, sys: Partition) -> ScoreHolder:
    """
    tests whether score(gold, sys) = score(sys, gold)
    """
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(isclose(x, y))
    return ScoreHolder.apply(evaluate(gold, sys), intern, evaluate(sys, gold))


def singleton_test(gold: Partition) -> ScoreHolder:
    return evaluate(gold, singleton_partition(get_mentions(gold)))


def entity_test(gold: Partition) -> ScoreHolder:
    return evaluate(gold, entity_partition(get_mentions(gold)))


def non_identity_test(gold: Partition, sys: Partition) -> Optional[ScoreHolder]:
    """
    tests whether score(gold, sys) != 1 with gold != sys
    """
    def intern(x: float):
        return BinaryResult.get_binary_result(not isclose(x, 1))
    if gold == sys:
        return None
    return evaluate(gold, sys).apply_to_values(intern)


def identity_test(gold: Partition) -> ScoreHolder:
    """
    tests wether score(gold, gold) =1
    """
    def intern(x: float):
        return BinaryResult.get_binary_result(isclose(x, 1))
    return evaluate(gold, gold).apply_to_values(intern)


def distance_triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder:
    def intern(x: float, y: float) -> BinaryResult:
        return BinaryResult.get_binary_result(isclose(x, y) or x < y)
    return ScoreHolder.apply(evaluate(a, c), intern, evaluate(a, b) + evaluate(b, c))


def triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder:
    def intern(x: float, y:float) -> BinaryResult:
        return BinaryResult.get_binary_result(isclose(x, y) or x > y)
    return ScoreHolder.apply(evaluate(b, b) + evaluate(a, c), intern, evaluate(a, b) + evaluate(b, c))


def randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    def intern(m) -> Iterator[ScoreHolder]:
        n = len(signature(test_func).parameters)
        for _ in range(repetitions):
            if partition_generators is None:
                a = (beta_partition(*beta_param, m) for _ in range(n))
            else:
                if len(partition_generators) != n:
                    # TODO exception
                    raise Exception(f'got {len(partition_generators)} partition generator, expected {n}')
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
        n = len(signature(test_func).parameters) - 1
        for gold in it:
            m = get_mentions(gold)
            for _ in range(repetitions):
                if partition_generators is None:
                    a = (beta_partition(*beta_param, m) for _ in range(n))
                else:
                    if len(partition_generators) != n:
                        raise Exception(f'got {len(partition_generators)} partition generator, expected {n}')
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
