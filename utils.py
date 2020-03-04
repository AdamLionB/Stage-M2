from __future__ import annotations

# std libs
from typing import Tuple, Dict, Iterator, Union, Any, Callable
from enum import Enum, auto
from math import isclose
from time import time
from itertools import zip_longest

# other libs
from sklearn import metrics

# scorch lib
from scorch.main import METRICS
from scorch.scores import conll2012

# this lib
from partition_utils import partition_to_sklearn_format, Partition
from scorer import lea, edit


class Timer:
    def __init__(self):
        self.dic = {}

    def timed(self, func: Callable) -> Callable:
        self.dic[func] = 0

        def intern(*args, **kwargs):
            start = time()
            res = func(*args, **kwargs)
            self.dic[func] += time() - start
            return res
        return intern

    def __repr__(self):
        res = ''
        for k, v in self.dic.items():
            res += f'{k.__name__}\t\t: {v}\n'
        return res


timer = Timer()

METRICS = {k: timer.timed(v) for k, v in METRICS.items()}


METRICS['conll'] = timer.timed(conll2012)
METRICS['LEA'] = timer.timed(lea)
# METRICS['edit'] = timer.timed(edit)
SK_METRICS = {
    'ARI': metrics.adjusted_rand_score,
    'HCV': metrics.homogeneity_completeness_v_measure,
    'AMI': metrics.adjusted_mutual_info_score,
    'FM': metrics.fowlkes_mallows_score
}


# TODO rename ?
class ScoreHolder:
    """
    A structure holding scores in a dict.
    Each key of the dic is the name of a metric,
    the tuples associated to each key are the values returned by the metric.
    The order in which the metrics are computed and thus added in the dictionnary is important
    since in python dictionnary views are in fact ordered. (kinda)
    """

    def __init__(self, dic: Dict[str, Tuple[Any, ...]]):
        self.dic = dic

    def __getitem__(self, item: str) -> Tuple[Any, ...]:
        return self.dic[item]

    def __eq__(self, other: ScoreHolder) -> bool:
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

    def __add__(self, other: ScoreHolder) -> ScoreHolder:
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
        return ScoreHolder({
            sk: tuple((x + y for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def __sub__(self, other: ScoreHolder) -> ScoreHolder:
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
        return ScoreHolder({
            sk: tuple((x - y for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def __mul__(self, scalar: float) -> ScoreHolder:
        """
        Outputs the result of the multiplication of a ScoreHolder by a scalar.
        Each value of the ScoreHolder is multiplied by the scalar.
        """
        return ScoreHolder({
            k: tuple((x * scalar for x in v))
            for k, v in self.dic.items()
        })

    def __truediv__(self, scalar: float) -> ScoreHolder:
        """
        Outputs the result of the division of a ScoreHolder by a scalar.
        Eeach value of the ScoreHolder is divided by the scalar.
        """
        return ScoreHolder({
            k: tuple((x / scalar for x in v))
            for k, v in self.dic.items()
        })

    def __pow__(self, power: float, modulo=None):
        """
        Outputs the result of raising a ScoreHolder to a given power.
        Each value of the ScoreHolder is raised to the given power.
        """
        return ScoreHolder({
            k: tuple((x ** power for x in v))
            for k, v in self.dic.items()
        })

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

    def compare(self, other: ScoreHolder) -> ScoreHolder:
        """
        Outputs the result of the comparison of a ScoreHolder to another.
        For two ScoreHolder to be compared they have to have the same structure,
        meanings the same keys, in the same order and tuples of similar size for each key
        """
        return ScoreHolder({
            sk: tuple((Growth.compare(x, y) for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    # TODO remove ? usefull ?
    def compare_t(self, t: Tuple) -> ScoreHolder:
        """
        Compare all tuple of the ScoreHolder to a tuple
        """
        return ScoreHolder({
            k: tuple(reversed(tuple((Growth.compare(x, y) for x, y in zip(v[::-1], t[::-1])))))
            for k, v in self.dic.items()
        })

    def apply(self, func: Callable[[Any, Any], Any], other: ScoreHolder) -> ScoreHolder:
        """
        Outputs the result of the comparison of a ScoreHolder to another.
        For two ScoreHolder to be compared they have to have the same structure,
        meanings the same keys, in the same order and tuples of similar size for each key
        """
        return ScoreHolder({
            sk: tuple((func(x, y) for x, y in zip(sv, ov)))
            for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
        })

    def apply_to_values(self, func: Callable[[Any], Any]) -> ScoreHolder:
        return ScoreHolder({
            k: tuple(func(x) for x in v)
            for k, v in self.dic.items()
        })

    def for_all_values(self) -> Iterator:
        for v in self.dic.values():
            for x in v:
                yield x

# TODO remove ? usefull ?
class Growth(Enum):
    """
    Enum used to compare function growth
    """
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
        """
        Identity function, it's defined just so ScoreHolder.average can be reused
        """
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
        if isclose(a, b):
            return Growth.CONST
        if a > b:
            return Growth.STRICT_DECR
        if a < b:
            return Growth.STRICT_INCR


class BinaryResult:
    """
    Allows easy aggregation and printing of test results
    """

    def __truediv__(self, other: Any):
        """
        Identity function, it's defined just so ScoreHolder.average can be reused
        """
        return self

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.value:
            return 'V'
        else:
            return 'X'


class EasyFail(BinaryResult, Enum):
    FAILED = False
    PASSED = True

    def __add__(self, other: EasyFail) -> EasyFail:
        if self.value & other.value:
            return EasyFail.PASSED
        else:
            return EasyFail.FAILED

    @staticmethod
    def has_passed_test(has_passed: bool) -> EasyFail:
        """
        Converts a boolean in an EasyFail BinaryResult
        """
        if has_passed:
            return EasyFail.PASSED
        else:
            return EasyFail.FAILED


class HardFail(BinaryResult, Enum):
    FAILED = False
    PASSED = True

    def __add__(self, other: HardFail) -> HardFail:
        if self.value | other.value:
            return HardFail.PASSED
        else:
            return HardFail.FAILED

    @staticmethod
    def has_passed_test(has_passed: bool) -> HardFail:
        """
        Converts a boolean in an HardFail BinaryResult
        """
        if has_passed:
            return HardFail.PASSED
        else:
            return HardFail.FAILED


def to_tuple(e: Union[Any, Tuple[Any]]) -> Tuple[Any]:
    """
    Output the input as tuple in a pure way.
    If the input was a tuple, return it. If it wasn't, puts it in a tuple then return it.
    """
    if type(e) == tuple:
        return e
    return e,


def evaluates(gold: Partition, sys: Partition) -> ScoreHolder:
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
