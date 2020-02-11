from __future__ import annotations
# std lib
from typing import Callable, Tuple, Union, Iterator, Optional
from inspect import signature
from os import listdir
from functools import partial

#scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions
from find_better_name import ScoreHolder, evaluate


def symetry_test(gold: Partition, sys: Partition) -> ScoreHolder:
    scores_a = evaluate(gold, sys)
    scores_b = evaluate(sys, gold)
    return scores_a.compare(scores_b)


def singleton_test(gold: Partition) -> ScoreHolder:
    return evaluate(gold, singleton_partition(get_mentions(gold)))


def entity_test(gold: Partition) -> ScoreHolder:
    return evaluate(gold, entity_partition(get_mentions(gold)))


def identity_test(gold: Partition) -> ScoreHolder:
    return evaluate(gold, gold)


def triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder:
    return ScoreHolder.compare(evaluate(a, c), evaluate(a, b) + evaluate(b, c))


# TODO rename
def r_test(
        test: Callable[[Partition, ...], ScoreHolder],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    def intern(m) -> Iterator[ScoreHolder]:
        n = len(signature(test).parameters)
        for _ in range(repetitions):
            if partition_generators is None:
                a = (beta_partition(*beta_param, m) for _ in range(n))
            else:
                if len(partition_generators) != n:
                    # TODO exception
                    raise Exception()
                a = (part() for part in partition_generators)
            yield test(*a)
    mentions = list(range(100))
    if std:
        return ScoreHolder.avg_std(intern(mentions))
    else:
        return ScoreHolder.average(intern(mentions))


def fixed_gold_test(
        test: Callable[[Partition, ...], ScoreHolder],
        it: Iterator[Partition],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    def intern() -> Iterator[ScoreHolder]:
        n = len(signature(test).parameters)-1
        for gold in it:
            m = get_mentions(gold)
            for _ in range(repetitions):
                if partition_generators is None:
                    a = (beta_partition(*beta_param, m) for _ in range(n))
                else:
                    if len(partition_generators) != n:
                        # TODO exception
                        raise Exception()
                    a = (part() for part in partition_generators)
            yield test(gold, *a)
    if std:
        return ScoreHolder.avg_std(intern())
    else:
        return ScoreHolder.average(intern())


def ancor_test(
        test: Callable[[Partition], ScoreHolder],
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Union[Tuple[ScoreHolder, ScoreHolder], ScoreHolder]:
    return fixed_gold_test(test, iter_ancor(), std=std, repetitions=repetitions, beta_param=beta_param)


# TODO generalise for any json format
# TODO generalise for any format ? (with a given read method)
def iter_ancor() -> Iterator[Partition]:
    """
    Iterates over the ancor corpus
    """
    for n, file in enumerate(listdir('../ancor/json/')):
        # if n > 0 : break
        yield clusters_from_json(open('../ancor/json/' + file))
