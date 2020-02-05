from __future__ import annotations

# std libs
from typing import Tuple, Dict, Iterator, Union, Any
from enum import Enum, auto

# other libs
from sklearn import metrics

# scorch lib
from scorch.main import METRICS
from scorch.scores import conll2012

# this lib
from partition_utils import partition_to_sklearn_format, Partition


METRICS['conll'] = conll2012
SK_METRICS = {
    'ARI': metrics.adjusted_rand_score,
    'HCV': metrics.homogeneity_completeness_v_measure,
    'AMI': metrics.adjusted_mutual_info_score,
    'FM': metrics.fowlkes_mallows_score
}

# TODO rename ?
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
                if issubclass(type(e), float):
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


def evaluate(gold: Partition, sys: Partition) -> Scores:
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
    return Scores(res)