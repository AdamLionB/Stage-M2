from __future__ import annotations

# std lib
from typing import Callable, Tuple, Union, Iterator, Optional
from inspect import signature
from os import listdir
from itertools import product
from math import isclose
from functools import partial

# scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions, \
    all_partition_of_size
from utils import ScoreHolder, evaluates, EasyFail, HardFail, to_tuple


def a1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, y) and x > y)

    gold = [{1, 2}, {3}, {4}]
    sys = [{1, 2}, {3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, y) and x > y)

    gold = [{3}, {4}]
    sys = [{3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a3_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, y) and x > y)

    gold = [{1, 2}, {3}]
    sys = [{1, 2, 3}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def b1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, y) and x < y)

    gold = [{1, 2}, {3, 4, 5}]
    sys1 = [{1, 2, 3}, {4, 5}]
    sys2 = [{1, 2, 3, 4, 5}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def b2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, y) and x < y)

    gold = [{1, 2, 3, 4, 5}, {6, 7}]
    sys1 = [{1, 2, 3, 4}, {5, 6, 7}]
    sys2 = [{1, 2}, {3, 4, 5}, {6, 7}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def d1_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(isclose(x, y))

    gold1 = [{1, 2}, {3, 4}, {5, 6}]
    sys1 = [{1, 3}, {2, 4}, {5, 6}]
    gold2 = [{1, 2}, {3, 4}, {5, 6, 7, 8}]
    sys2 = [{1, 3}, {2, 4}, {5, 6, 7, 8}]
    return ScoreHolder.apply(evaluates(gold1, sys1), intern, evaluates(gold2, sys2))


def d2_test() -> ScoreHolder:
    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(isclose(x, y))

    gold = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys1 = [{1, 2, 3, 4, 6}, {5, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys2 = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 14}, {13, 15, 16}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def f_test(gold: Partition, sys: Partition) -> ScoreHolder:
    def intern(x: float) -> HardFail:
        return HardFail.has_passed_test(isclose(x, 0))

    return ScoreHolder.apply_to_values(evaluates(gold, sys), intern)


def symetry_test(gold: Partition, sys: Partition) -> ScoreHolder:
    """
    evaluates whether score(gold, sys) = score(sys, gold)
    """

    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(isclose(x, y))

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

    def intern(x: float) -> EasyFail:
        return EasyFail.has_passed_test(not isclose(x, 1))

    if gold == sys:
        return None
    return evaluates(gold, sys).apply_to_values(intern)


def identity_test(gold: Partition) -> ScoreHolder:
    """
    evaluates whether score(gold, gold) = 1
    """

    def intern(x: float) -> EasyFail:
        return EasyFail.has_passed_test(isclose(x, 1))

    return evaluates(gold, gold).apply_to_values(intern)


def triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder:
    """
    evaluates whether the (similarity) triangle inequality is respected
    s(a,b) + s(b,c) <= s(b,b) + s(a,b)
    """

    def intern(x: float, y: float) -> EasyFail:
        return EasyFail.has_passed_test(isclose(x, y) or x < y)

    return ScoreHolder.apply(evaluates(a, b) + evaluates(b, c), intern, evaluates(b, b) + evaluates(a, c))


def randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[ScoreHolder]:
    """
    Apply the given test function to partitions randomly generated with partition_generators
    """
    n_args = len(signature(test_func).parameters)
    m = list(range(100))
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


def fixed_gold_randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder]],
        it: Iterator[Partition],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[ScoreHolder]:
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


def ancor_gold_randomized_test(
        test_func: Callable[[Partition], Optional[ScoreHolder]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[ScoreHolder]:
    return fixed_gold_randomized_test(test_func, iter_ancor(), partition_generators=partition_generators, std=std,
                                      repetitions=repetitions, beta_param=beta_param)


def all_partitions_test(
        test_func: Callable[[Partition], Optional[ScoreHolder]],
        i: int = 1
) -> Iterator[ScoreHolder]:
    n_args = len(signature(test_func).parameters)
    for args in product(all_partition_of_size(i), repeat=n_args):
        res = test_func(*args)
        if res is not None:
            yield res


class tmp_class:
    distributions = [partial(beta_partition, a=1, b=1), partial(beta_partition, a=1, b=100)]

    def __init__(
            self,
            test_func: Callable[[Partition, ...], ScoreHolder],
            descr_str: str,
            on_corpus: bool = False,
            repetitions: int = 100,
            std: bool = False,
            start: int = 1,
            end: int = 5
    ):
        self.test_func = test_func
        self.n_args = len(signature(self.test_func).parameters)
        self.descr_str = descr_str
        self.on_corpus = on_corpus
        self.repetitions = repetitions
        self.std = std
        self.start = start
        self.end = end

    @staticmethod
    def avg_tuples(score_holderss: Iterator[Tuple[ScoreHolder, ...]]) -> Tuple[ScoreHolder, ...]:
        """
        Outputs the average ScoreHolder of an iterator.
        All of the ScoreHolder in the iterator have to have the same strucutre,
        meanings the same keys, in the same order and tuples of similar size for each key
        """
        ress = list(next(score_holderss))
        count = 1
        for score_holders in score_holderss:
            for n, score_holder in enumerate(score_holders):
                ress[n] += score_holder
            count += 1
        for n, score_holder in enumerate(ress):
            ress[n] /= count
        return tuple(ress)

    @staticmethod
    def average(scoress: Iterator[ScoreHolder]) -> ScoreHolder:
        """
        Outputs the average ScoreHolder of an iterator.
        All of the ScoreHolder in the iterator have to have the same strucutre,
        meanings the same keys, in the same order and tuples of similar size for each key
        """
        res = next(scoress)
        count = 1
        for scores in scoress:
            res += scores
            count += 1
        return res / count

    @staticmethod
    def avg_std(scoress: Iterator[ScoreHolder]) -> Tuple[ScoreHolder, ScoreHolder]:
        """
        Outputs the average and standard deviation ScoreHolder of an iterator.
        All of the ScoreHolder in the iterator have to have the same strucutre,
        meanings the same keys, in the same order and tuples of similar size for each key
        """
        regular_sum = next(scoress)
        squared_sum = regular_sum ** 2
        count = 1
        for scores in scoress:
            regular_sum += scores
            squared_sum += scores ** 2
            count += 1
        return (regular_sum / count,
                ((squared_sum - (regular_sum ** 2) / count) / count) ** (1 / 2))


    def agreg(self, it: Iterator[ScoreHolder]) -> Iterator[ScoreHolder]:
        if self.std:
            return tmp_class.avg_std(it)
        else:
            return tmp_class.average(it)

    def intern1(self) -> Iterator[ScoreHolder]:
        for i in range(self.start, self.end):
            yield self.agreg(all_partitions_test(self.test_func, i=i))

    def intern2(self) -> Iterator[ScoreHolder]:
        for dists in product(tmp_class.distributions, repeat=self.n_args):
            yield self.agreg(randomized_test(self.test_func, partition_generators=dists, repetitions=self.repetitions))

    def intern3(self, repeat) -> Iterator[ScoreHolder]:
        for dists in product(tmp_class.distributions, repeat=repeat):
            yield self.agreg(ancor_gold_randomized_test(self.test_func, partition_generators=dists))

    def f(self):
        if self.n_args != 0:
            yield tmp_class.avg_tuples(map(to_tuple, self.intern1()))
            if self.on_corpus:
                yield tmp_class.avg_tuples(map(to_tuple, self.intern3(self.repetitions if self.n_args != 1 else 1)))
        yield tmp_class.avg_tuples(map(to_tuple, self.intern2()))

    def g(self):
        print(self.descr_str)
        for x in tmp_class.avg_tuples(self.f()):
            print(x)
        print('-------------')

    def h(self):
        map(to_tuple, self.intern1())


ALL_TESTS = {
    a1_test: tmp_class(a1_test, 'a1', repetitions=1),
    a2_test: tmp_class(a2_test, 'a2', repetitions=1),
    a3_test: tmp_class(a3_test, 'a3', repetitions=1),
    b1_test: tmp_class(b1_test, 'b1', repetitions=1),
    b2_test: tmp_class(b2_test, 'b2', repetitions=1),
    d1_test: tmp_class(d1_test, 'd1', repetitions=1),
    d2_test: tmp_class(d2_test, 'd2', repetitions=1),
    identity_test: tmp_class(identity_test, 'e1 | identity', repetitions=100),
    non_identity_test: tmp_class(non_identity_test, 'e2 | non_identity', repetitions=100, start=2),
    f_test: tmp_class(f_test, 'f', repetitions=100),
    triangle_test: tmp_class(triangle_test, 'g | triangle', repetitions=100),
    symetry_test: tmp_class(symetry_test, 'h | symetry', repetitions=100),
    singleton_test: tmp_class(singleton_test, 'singleton', repetitions=100),
    entity_test: tmp_class(entity_test, 'entity', repetitions=100),
}


# TODO generalise for any json format
# TODO generalise for any format ? (with a given read method)
def iter_ancor() -> Iterator[Partition]:
    """
    Iterates over the ancor corpus
    """
    for n, file in enumerate(listdir('../ancor/json/')):
        # if n > 0 : break
        yield clusters_from_json(open('../ancor/json/' + file))
