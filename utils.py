from __future__ import annotations

# std libs
from typing import Tuple, Dict, Iterator, Union, Any, Callable, TypeVar, List, Generic
from enum import Enum, auto
from math import isclose
from time import time
from functools import reduce, partial
from itertools import zip_longest

# other libs
from sklearn import metrics

# scorch lib
from scorch.main import METRICS
from scorch.scores import conll2012

# this lib
from partition_utils import partition_to_sklearn_format, Partition
from scorer import lea, edit

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

METRICS['conll'] = conll2012
METRICS['LEA'] = lea
# METRICS['edit'] = edit
SK_METRICS = {
    'ARI': metrics.adjusted_rand_score,
    'HCV': metrics.homogeneity_completeness_v_measure,
    'AMI': metrics.adjusted_mutual_info_score,
    'FM': metrics.fowlkes_mallows_score
}


class ScoreHolder(Generic[T]):
    """
    A structure holding scores in a dict.
    Each key of the dic is the name of a metric,
    the tuples associated to each key are the values returned by the metric.
    The order in which the metrics are computed and thus added in the dictionnary is important
    since in python dictionnary views are in fact ordered. (kinda)
    """

    def __init__(self, dic: Dict[str, Tuple[T, ...]]):
        self.dic = dic

    def __getitem__(self, item: str) -> Tuple[T, ...]:
        return self.dic[item]

    def __eq__(self, other: ScoreHolder[U]) -> bool:
        """
        Checks if two ScoreHolder are equal.
        To be considered equals two ScoreHolder have to have:
        The same keys in the same order and the same tuple for each keys
        """
        if len(self.dic) != len(other.dic):
            return False
        for (sk, sv), (ok, ov) in zip(self.dic.items(), other.dic.items()):
            if sk != ok:
                return False
            if len(sv) != len(ov):
                return False
            for x, y in zip(sv, ov):
                if x != y:
                    return False
        return True

    def __add__(self, other: ScoreHolder[T]) -> ScoreHolder[T]:
        """
        Outputs the result of the addition of two ScoreHolder.
        For two ScoreHolder to be added they have to have the same structure,
        meanings the same keys, in the same order and tuples of similar size for each key

        Example:

        >>> ScoreHolder({'a': (1.0,), 'b': (1.0, 2.0)}) + ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        is valid
        >>> ScoreHolder({'a': (1.0,), 'b': (1.0,)}) + ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        isn't
        >>> ScoreHolder({'a': (1.0,), 'c': (1.0, 2.0)}) + ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        isn't
        """
        return self.apply(lambda x, y : x + y, other)

    def __sub__(self, other: ScoreHolder[T]) -> ScoreHolder[T]:
        """
        Outputs the result of the substraction of a ScoreHolder by another.
        For two ScoreHolder to be substracted they have to have the same structure,
        meanings the same keys, in the same order and tuples of similar size for each key

        Example:

        >>> ScoreHolder({'a': (1.0,), 'b': (1.0, 2.0)}) - ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        is valid
        >>> ScoreHolder({'a': (1.0,), 'b': (1.0,)}) - ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        isn't
        >>> ScoreHolder({'a': (1.0,), 'c': (1.0, 2.0)}) - ScoreHolder({'a': (3.0,), 'b': (4.0, 5.0)})
        isn't
        """
        return self.apply(lambda x, y: x - y, other)

    def __mul__(self, scalar: float) -> ScoreHolder[T]:
        """
        Outputs the result of the multiplication of a ScoreHolder by a scalar.
        Each value of the ScoreHolder is multiplied by the scalar.
        """
        return self.apply_to_values(lambda x : x * scalar)

    def __truediv__(self, scalar: float) -> ScoreHolder[T]:
        """
        Outputs the result of the division of a ScoreHolder by a scalar.
        Eeach value of the ScoreHolder is divided by the scalar.
        """
        return self.apply_to_values(lambda x: x / scalar)

    def __pow__(self, power: float, modulo=None) -> ScoreHolder[T]:
        """
        Outputs the result of raising a ScoreHolder to a given power.
        Each value of the ScoreHolder is raised to the given power.
        """
        return self.apply_to_values(lambda x: x ** power)

    def __str__(self) -> str:
        res = '\n\t\tF\tP\tR'
        for k, v in self.dic.items():
            res += f'\n{k}\t:'
            for e in reversed(v):
                if issubclass(type(e), float):
                    res += f'\t{e:.2f}'
                else:
                    res += f'\t{e}'
        return res

    def __repr__(self) -> str:
        return str(self)

    def apply(self, func: Callable[[T, U], V], other: ScoreHolder[U]) -> ScoreHolder[V]:
        """
        Apply func to each pair of corresponding values of the two ScoreHolder and returns the result
        """
        return ScoreHolder({
            sk: tuple((func(x, y) for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def apply_to_values(self, func: Callable[[T], U]) -> ScoreHolder[U]:
        """
        Apply func to each value of the ScoreHolder and return the result
        """
        return ScoreHolder({
            k: tuple(func(x) for x in v)
            for k, v in self.dic.items()
        })

    # TODO remove ?
    def for_all_values(self) -> Iterator[T]:
        for v in self.dic.values():
            for x in v:
                yield x

    def for_important_values(self) -> Iterator[T]:
        """
        Iterates through the the last (most important) values of each scores
        """
        for v in self.dic.values():
            yield v[-1]


def to_tuple(e: Union[T, Tuple[T]]) -> Tuple[T]:
    """
    Output the input as tuple in a pure way.
    If the input was a tuple, return it. If it wasn't, puts it in a tuple then return it.
    """
    if type(e) == tuple:
        return e
    return e,


def evaluates(gold: Partition, sys: Partition) -> ScoreHolder[float]:
    """
    Computes metrics scores for a (gold, sys) and outputs it as a Scores
    """
    res = {}
    for name, metric in METRICS.items():
        res[name] = to_tuple(metric(gold, sys))
    gold = partition_to_sklearn_format(gold)
    sys = partition_to_sklearn_format(sys)
    if len(gold) == len(sys):
        for name, metric in SK_METRICS.items():
            res[name] = to_tuple(metric(gold, sys))
    return ScoreHolder(res)


def identity(x: T) -> T:
    return x


def first(x: T) -> U:
    return x[0]


def second(x: T) -> U:
    return x[1]


def acc(
        init_func: Callable[[T], U],
        acc1: Callable[[U, T], U],
        acc2: Callable[[U], V],
        scoress: Iterator[T]
) -> V:
    """
    ughh.. hard to describe, some kind of big reducer
    Takes an init function, two reduce function and an iterator
    applys the init function to the first element of the iterator then uses the first reduce on the reste of the iterator
    then apply the second reduce on the result of the last reduce.
    """
    res = init_func(next(scoress))
    for scores in scoress:
        res = acc1(res, scores)
    return acc2(res)


def list_and_second(x: Tuple[Any, ScoreHolder[bool]]) -> Tuple[List, ScoreHolder[bool]]:
    if not reduce(lambda a, b: a & b, x[1].for_important_values(), True):
        return [x[0]], x[1]
    else :
        return [], x[1]


def list_or_second(x: Tuple[Any, ScoreHolder[bool]]) -> Tuple[List, ScoreHolder[bool]]:
    if reduce(lambda a, b: a | b, x[1].for_important_values(), True):
        return [x[0]], x[1]
    else :
        return [], x[1]


def list_and(x: Tuple[List, ScoreHolder[bool]], y: Tuple[Any, ScoreHolder[bool]]) -> Tuple[List, ScoreHolder[bool]]:
    res = ScoreHolder.apply(x[1], lambda a, b: a & b, y[1])
    if reduce(lambda x, y: x | y,ScoreHolder.apply(x[1],lambda a, b: ((not b) and a), res).for_important_values()):
        return x[0]+[y[0]], res
    return x[0], res


def list_or(x: Tuple[List, ScoreHolder[bool]], y: Tuple[Any, ScoreHolder[bool]]) -> Tuple[List, ScoreHolder[bool]]:
    res = ScoreHolder.apply(x[1], lambda a, b: a | b, y[1])
    if reduce(lambda x, y: x | y,ScoreHolder.apply(x[1],lambda a, b: ((not a) and b), res).for_important_values()):
        return x[0]+[y[0]], res
    return x[0], res


def simple_and_acc(scoress: Iterator[Tuple[Any, ScoreHolder[bool]]]) -> ScoreHolder[bool]:
    return acc(second, lambda x, y: ScoreHolder.apply(x, lambda a, b: a & b, second(y)), identity, scoress)


def simple_or_acc(scoress: Iterator[Tuple[Any, ScoreHolder[bool]]]) -> ScoreHolder[bool]:
    return acc(second, lambda x, y: ScoreHolder.apply(x, lambda a, b : a | b ,second(y)), identity, scoress)


def list_and_acc(scoress: Iterator[Tuple[Any, ScoreHolder[bool]]]) -> Tuple[list, ScoreHolder[bool]]:
    return acc(list_and_second, list_and, identity, scoress)


def list_or_acc(scoress: Iterator[Tuple[Any, ScoreHolder[bool]]]) -> Tuple[list, ScoreHolder[bool]]:
    return acc(list_or_second, list_or, identity, scoress)


def list_and_acc_acc(scoress: Iterator[Tuple[list, ScoreHolder[bool]]]) -> Tuple[list, ScoreHolder[bool]]:
    return acc(
        lambda x: ([x[0], x[1][0]], x[1][1]),
        lambda x, y: (x[0] + [(y[0], y[1][0])], ScoreHolder.apply(x[1], lambda a, b: a & b, y[1][1])),
        identity,
        scoress
    )


def list_or_acc_acc(scoress: Iterator[Tuple[list, ScoreHolder[bool]]]) -> Tuple[list, ScoreHolder[bool]]:
    return acc(
        lambda x: ([x[0], x[1][0]], x[1][1]),
        lambda x, y: (x[0]+[(y[0], y[1][0])], ScoreHolder.apply(x[1], lambda a, b : a | b, y[1][1])),
        identity,
        scoress
    )
