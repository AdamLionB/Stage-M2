from __future__ import annotations

# std lib
from typing import Callable, Tuple, Union, Iterator, Optional, Any, TypeVar, List
from inspect import signature
from os import listdir
from itertools import product
from math import isclose
from functools import partial, reduce

# other lib
import numpy as np
import matplotlib.pyplot as plt

# scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions, \
    all_partition_of_size, is_regular, contains_singleton, introduce_randomness
from utils import ScoreHolder, evaluates, evaluates_main, to_tuple, simple_and_acc, simple_or_acc,\
    list_and_acc1, list_and_acc2, list_or_acc1, list_or_acc2, micro_avg_acc1, micro_avg_acc2, \
    micro_avg_std_acc1, micro_avg_std_acc2, macro_avg_acc, macro_avg_std_acc1, macro_avg_std_acc2, series_micro_acc1, \
    series_micro_acc2

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

def a1_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x > y

    gold = [{1, 2}, {3}, {4}]
    sys = [{1, 2}, {3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a2_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x > y

    gold = [{3}, {4}]
    sys = [{3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def a3_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x > y

    gold = [{1, 2}, {3}]
    sys = [{1, 2, 3}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def b1_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x < y

    gold = [{1, 2}, {3, 4, 5}]
    sys1 = [{1, 2, 3}, {4, 5}]
    sys2 = [{1, 2, 3, 4, 5}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def b2_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x < y

    gold = [{1, 2, 3, 4, 5}, {6, 7}]
    sys1 = [{1, 2, 3, 4}, {5, 6, 7}]
    sys2 = [{1, 2}, {3, 4, 5}, {6, 7}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def c_test(gold: Partition, *, modifications: int = 1) -> ScoreHolder[float]:
    sys = gold
    for i in range(modifications):
        sys= introduce_randomness(sys)
    return evaluates_main(gold, sys)

def d1_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return isclose(x, y)

    gold1 = [{1, 2}, {3, 4}, {5, 6}]
    sys1 = [{1, 3}, {2, 4}, {5, 6}]
    gold2 = [{1, 2}, {3, 4}, {5, 6, 7, 8}]
    sys2 = [{1, 3}, {2, 4}, {5, 6, 7, 8}]
    return ScoreHolder.apply(evaluates(gold1, sys1), intern, evaluates(gold2, sys2))


def d2_test() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return isclose(x, y)

    gold = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys1 = [{1, 2, 3, 4, 6}, {5, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys2 = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 14}, {13, 15, 16}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def f_test(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern(x: float) -> bool:
        return isclose(x, 0)

    return ScoreHolder.apply_to_values(evaluates(gold, sys), intern)





def singleton_test(gold: Partition) -> ScoreHolder[float]:
    """
    evaluates the score of a partition against a partition composed only of singletons
    """
    return evaluates(gold, singleton_partition(get_mentions(gold)))


def entity_test(gold: Partition) -> ScoreHolder[float]:
    """
    evaluates the score of a partition against a partition composed of only one entity
    """
    return evaluates(gold, entity_partition(get_mentions(gold)))


def non_identity_test(gold: Partition, sys: Partition) -> Optional[ScoreHolder[bool]]:
    """
    evaluates whether score(gold, sys) != 1 with gold != sys
    """

    def intern(x: float) -> bool:
        return not isclose(x, 1)
        #return EasyFail.has_passed_test(not isclose(x, 1))

    if gold == sys:
        return None
    return evaluates(gold, sys).apply_to_values(intern)


def identity_test(gold: Partition) -> ScoreHolder[bool]:
    """
    evaluates whether score(gold, gold) = 1
    """

    def intern(x: float) -> bool:
        return isclose(x, 1)

    return evaluates(gold, gold).apply_to_values(intern)


def metric_1_symetry_test(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    """
    evaluates whether score(gold, sys) = score(sys, gold)
    """

    def intern(x: float, y: float) -> bool:
        return isclose(x, y)

    return ScoreHolder.apply(evaluates(gold, sys), intern, evaluates(sys, gold))


def metric_2_non_negativity_test(gold: Partition) -> ScoreHolder[bool]:
    def intern(x: float) -> bool:
        return isclose(x, 0) or x > 0

    return evaluates(gold, gold).apply_to_values(intern)


def metric_3(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern(x: float, y:float) -> bool:
        return isclose(x, y) or x > y

    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def metric_4_triangle_test(a: Partition, b: Partition, c: Partition) -> ScoreHolder[bool]:
    """
    evaluates whether the (similarity) triangle inequality is respected
    s(a,b) + s(b,c) <= s(b,b) + s(a,b)
    """

    def intern(x: float, y: float) -> bool:
        return isclose(x, y) or x < y

    return ScoreHolder.apply(evaluates(a, b) + evaluates(b, c), intern, evaluates(b, b) + evaluates(a, c))

def metric_5_indiscernable(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern1(x: float, y: float):
        return isclose(x, y)

    def intern2(x: bool, y:bool):
        return x & y

    a = evaluates(gold, gold)
    b = evaluates(sys, sys)
    c = evaluates(gold, sys)

    x = ScoreHolder.apply(a, intern1, b)
    y = ScoreHolder.apply(a, intern1, c)
    z = ScoreHolder.apply(b, intern1, c)

    if gold == sys:
        return ScoreHolder.apply(
            ScoreHolder.apply(x, intern2, y),
            intern2,
            z
        )
    else:
        return ScoreHolder.apply(
            ScoreHolder.apply(x, intern2, y),
            intern2,
            z
        ).apply_to_values(lambda k: not k)


def metric_6(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern(x: float):
        return isclose(x, 1) or x < 1

    return evaluates(gold, sys).apply_to_values(intern)


def metric_7(gold: Partition) -> ScoreHolder[bool]:
    def intern(x: float):
        return isclose(x, 1)

    return evaluates(gold, gold).apply_to_values(intern)


def metric_8(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern(x: float):
        return isclose(x, 0) or x > 0

    return evaluates(gold, sys).apply_to_values(intern)


def randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder[T]]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[List[Partition], ScoreHolder[T]]:
    """
    Apply the given test function to partitions randomly generated with partition_generators
    """
    n_args = len(signature(test_func).parameters)
    m = list(range(100))
    # repeats the test 'repetitions' times
    for _ in range(repetitions):
        # if no partition_generator is given partition will follow a beta partition
        if partition_generators is None:
            partitions = [beta_partition(*beta_param, m) for _ in range(n_args)]
        else:
            if len(partition_generators) != n_args:
                raise Exception(f'got {len(partition_generators)} partition generator, expected {n_args}')
            partitions = [part(m) for part in partition_generators]
        res = test_func(*partitions)
        if res is not None:
            yield partitions, res



def fixed_gold_randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder[T]]],
        it: Iterator[Partition],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[List[Partition], ScoreHolder[T]]:
    n_agrs = len(signature(test_func).parameters) - 1
    for gold in it:
        m = get_mentions(gold)
        # repeats the test 'repetitions' times
        for _ in range(repetitions):
            # if no partition_generator is given partition will follow a beta partition
            if partition_generators is None:
                partitions = [beta_partition(*beta_param, m) for _ in range(n_agrs)]
            else:
                if len(partition_generators) != n_agrs:
                    raise Exception(f'got {len(partition_generators)} partition generator, expected {n_agrs}')
                partitions = [part(m) for part in partition_generators]
        res = test_func(gold, *partitions)
        if res is not None:
            yield partitions, res


def ancor_gold_randomized_test(
        test_func: Callable[[Partition], Optional[ScoreHolder[T]]],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        std: bool = False,
        repetitions: int = 100,
        beta_param: Tuple[float, float] = (1, 1)
) -> Iterator[List[Partition], ScoreHolder[T]]:
    return fixed_gold_randomized_test(test_func, iter_ancor(), partition_generators=partition_generators, std=std,
                                      repetitions=repetitions, beta_param=beta_param)


def variation_gold_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder[T]]],
        it: Iterator[Partition],
        repetitions: int = 100
):
    n_agrs = len(signature(test_func).parameters) - 1
    for gold in it:
        m = get_mentions(gold)
        # repeats the test 'repetitions' times
        for _ in range(repetitions):
            partitions = [introduce_randomness(gold) for n in range(n_agrs)]
            res = test_func(gold, *partitions)
            yield partitions, res



def all_partitions_test(
        test_func: Callable[[Partition], Optional[ScoreHolder[T]]],
        i: int = 1,
        filtre_func: Callable[[Partition], bool] = lambda x : True,
        repetitions: int = 1
) -> Iterator[Tuple[Partition], ScoreHolder[T]]:
    n_args = len(signature(test_func).parameters)
    for partitions in product(filter(filtre_func,all_partition_of_size(i)), repeat=n_args):
        for _ in range(repetitions):
            res = test_func(*partitions)
            if res is not None:
                yield partitions, res


from dataclasses import dataclass

@dataclass
class exp_setup:
    active: bool = False
    repetitions: int = 1
    start: int = 1
    end: int = 1
class tmp_class:
    distributions = [partial(beta_partition, a=1, b=1)]#, partial(beta_partition, a=1, b=100)]

    def __init__(
            self,
            test_func: Callable[[Partition, ...], ScoreHolder[T]],
            descr_str: str,
            agg: Tuple[Callable[[Iterator[T]], U], Callable[[Iterator[U]], V]] = (simple_and_acc, simple_and_acc),
            *,
            systematic: exp_setup = exp_setup(False, 1, 6, 7),
            on_corpus: exp_setup = exp_setup(False, 1, 6, 7),
            randomize: exp_setup = exp_setup(False, 1, 6, 7),
            variation: exp_setup = exp_setup(False, 30, 6, 7)
    ):
        self.test_func = test_func
        self.n_args = len(signature(self.test_func).parameters)
        self.descr_str = descr_str
        self.on_corpus = on_corpus
        self.systematic = systematic
        self.randomize = randomize
        self.variation = variation
        self.agg = agg

    def intern1(self) -> Iterator[Tuple[int, T]]:
        for i in range(self.systematic.start, self.systematic.end):
            yield i, self.agg[0](all_partitions_test(self.test_func, i=i, repetitions=self.systematic.repetitions))


    def intern2(self) -> Iterator[Tuple[int, T]]:
        for dists in product(tmp_class.distributions, repeat=self.n_args):
            yield dists, self.agg[0](randomized_test(self.test_func, partition_generators=dists, repetitions=self.randomize.repetitions))

    def intern3(self) -> Iterator[Tuple[int, T]]:
        for dists in product(tmp_class.distributions, repeat=self.n_args-1):
            yield dists, self.agg[0](
                ancor_gold_randomized_test(
                    self.test_func,
                    partition_generators=dists,
                    repetitions=self.on_corpus.repetitions
                )
            )

    # def intern4(self) -> Iterator[Tuple[int, T]]:
    #     for i in range(self.start, self.end):
    #         yield i, self.agg(variation_gold_test(self.test_func, it=all_partition_of_size(i), repetitions=self.repetitions))

    def f(self) -> Iterator[T]:
        if self.n_args != 0:
            if self.systematic.active:
                try :
                    yield self.agg[1](self.intern1())
                except StopIteration:
                    pass
            if self.on_corpus.active:
                yield self.agg[1](self.intern3())
            # if self.variation:
            #     yield self.agg2(self.intern4())
        if self.randomize.active:
            yield self.agg[1](self.intern2())

    def g2(self) -> None:
        print(self.descr_str)
        for x in self.f():
            print(x)
        print('-------------')


    def g3(self) -> None:
        fig = plt.plot()
        Y = {}
        E = {}
        X = []
        L = []
        for a in self.f():
            for k, (v1, v2) in a.items():
                X.append(k)
                if not L:
                    L = list(v1.dic.keys())
                for l, i, j in zip(L, v1.dic.values(), v2.dic.values()):
                    Y[l] = Y.setdefault(l, []) + [i[0]]
                    E[l] = E.setdefault(l, []) + [j[0]]
        for y, e, l in zip(Y.values(), E.values(), L):
            x, y, e = zip(*sorted(zip(X, y, e)))
            plt.errorbar(x, y, yerr=e, label=l)
        plt.ylim(0, 1)
        plt.xlim(0, 100)
        plt.legend()
        plt.show()

    def h(self):
        map(to_tuple, self.intern1())


ALL_TESTS = {
    a1_test: tmp_class(a1_test, 'a1', systematic=exp_setup(True, 1, 1, 7)),
    a2_test: tmp_class(a2_test, 'a2', systematic=exp_setup(True, 1, 1, 7)),
    a3_test: tmp_class(a3_test, 'a3', systematic=exp_setup(True, 1, 1, 7)),
    b1_test: tmp_class(b1_test, 'b1', systematic=exp_setup(True, 1, 1, 7)),
    b2_test: tmp_class(b2_test, 'b2', systematic=exp_setup(True, 1, 1, 7)),
    c_test: tmp_class(lambda x: c_test(x, modifications=4), 'c',
                      agg=(series_micro_acc1, series_micro_acc2), systematic=exp_setup(True, 30, 9, 10)),
    d1_test: tmp_class(d1_test, 'd1', systematic=exp_setup(True, 1, 1, 7)),
    d2_test: tmp_class(d2_test, 'd2', systematic=exp_setup(True, 1, 1, 7)),
    identity_test: tmp_class(identity_test, 'e1 | identity', randomize=exp_setup(True, 100, 1, 7)),
    non_identity_test: tmp_class(non_identity_test, 'e2 | non_identity', randomize=exp_setup(True, 100, 1, 7)),
    f_test: tmp_class(f_test, 'f', agg=(simple_or_acc, simple_or_acc), randomize=exp_setup(True, 100, 1, 7)),
    metric_1_symetry_test: tmp_class(metric_1_symetry_test, 'metrique 1',
                                     agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_2_non_negativity_test: tmp_class(metric_2_non_negativity_test, 'metrique 2',
                                            agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_3: tmp_class(metric_3, 'metrique 3',
                        agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_4_triangle_test: tmp_class(metric_4_triangle_test, 'metrique 4',
                                      agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 6)),
    metric_5_indiscernable: tmp_class(metric_5_indiscernable, 'metrique 5',
                                      agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_6: tmp_class(metric_6, 'metrique 6',
                        agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_7: tmp_class(metric_7, 'metrique 7',
                        agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    metric_8: tmp_class(metric_8, 'metrique 8',
                        agg=(list_and_acc1, list_and_acc2), randomize=exp_setup(True, 100, 1, 7)),
    singleton_test: tmp_class(singleton_test, 'singleton',
                              agg=(macro_avg_acc, macro_avg_acc), randomize=exp_setup(True, 100, 1, 7)),
    entity_test: tmp_class(entity_test, 'entity',
                           agg=(macro_avg_acc, macro_avg_acc), randomize=exp_setup(True, 100, 1, 7)),
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
