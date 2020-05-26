from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from itertools import product
from math import isclose
from os import listdir
# std lib
from typing import Callable, Tuple, Iterator, Optional, TypeVar, List

# other lib
import matplotlib.pyplot as plt
# scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions, \
    all_partition_of_size, introduce_randomness, r_part
from utils import ScoreHolder, evaluates, evaluates_main, to_tuple, simple_and_acc, simple_or_acc, \
    list_and_acc1, list_and_acc2, macro_avg_acc, series_micro_acc1, \
    series_micro_acc2

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


def sing_1_a1() -> ScoreHolder[bool]:
    """
    Runs the SING_1 a1 test which aims to check if the score recognize a correctly resolved singleton from a wrong one.
    To case are compared s([{1, 2}, {3}, {4}], [{1, 2}, {3}, {4}]) vs s([{1, 2}, {3}, {4}], [{1, 2}, {3, 4}])
    The entity {1, 2} is here to avoid the know issue of division of some score by 0 for fully dual partition. This
    entity is in both case perfectly resolved and should not impact the result of the test.
    The singleton {3} and {4} are in the first case perfectly resolved, and mistakenly fused in the second case.
    As the only difference between the two case are the mentions 3 and 4 then if the score correctly distinguish
    correctly resolved singleton  the first case should have a higher score than the second.
    """
    def intern(x: float, y: float) -> bool:
        """
        x >= y for floats
        """
        return not isclose(x, y) and x > y

    gold = [{1, 2}, {3}, {4}]
    sys = [{1, 2}, {3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def sing_1_a2() -> ScoreHolder[bool]:
    """
    Runs the SING_1 a2 test which aims to check if the score recognize a correctly resolved singleton from a wrong one.
    To case are compared s([{3}, {4}], [{3}, {4}]) vs s([{3}, {4}], [{3, 4}])
    Unlike the SING_1 a1 test which avoid fully dual partition by adding the entity {1, 2} to the partition, this test
    does not.
    The singleton {3} and {4} are in the first case perfectly resolved, and mistakenly fused in the second case.
    As the only difference between the two case are the mentions 3 and 4 then if the score correctly distinguish
    correctly resolved singleton  the first case should have a higher score than the second.
    """
    def intern(x: float, y: float) -> bool:
        """
            x >= y for floats
        """
        return not isclose(x, y) and x > y

    gold = [{3}, {4}]
    sys = [{3, 4}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def sing_1_b() -> ScoreHolder[bool]:
    """
    Runs the SING_1 b test which aims to check if the score recognize a correctly resolved singleton from a wrong one.
    To case are compared s([{1, 2}, {3}], [{1, 2}, {3}]) vs s([{1, 2}, {3}], [{1, 2, 3}])
    The entity {1, 2} contains 2 mentions in order to avoid the known issue of division of some score by 0 for
    fully dual partition a. This entity is in both case perfectly resolved and should not impact the result of the test.
    The singleton {3} and {4} are in the first case perfectly resolved, and mistakenly fused in the second case.
    As the only difference between the two case are the mentions 3 and 4 then if the score correctly distinguish
    correctly resolved singleton  the first case should have a higher score than the second.
    """
    #TODO
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x > y

    gold = [{1, 2}, {3}]
    sys = [{1, 2, 3}]
    return ScoreHolder.apply(evaluates(gold, gold), intern, evaluates(gold, sys))


def biais_1() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x < y

    gold = [{1, 2}, {3, 4, 5}]
    sys1 = [{1, 2, 3}, {4, 5}]
    sys2 = [{1, 2, 3, 4, 5}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def biais_2() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return not isclose(x, y) and x < y

    gold = [{1, 2, 3, 4, 5}, {6, 7}]
    sys1 = [{1, 2, 3, 4}, {5, 6, 7}]
    sys2 = [{1, 2}, {3, 4, 5}, {6, 7}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def sing_2(gold: Partition, *, modifications: int = 1) -> ScoreHolder[float]:
    sys = gold
    for i in range(modifications):
        sys = introduce_randomness(sys)
    return evaluates_main(gold, sys)


def weight_1() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return isclose(x, y)

    gold1 = [{1, 2}, {3, 4}, {5, 6}]
    sys1 = [{1, 3}, {2, 4}, {5, 6}]
    gold2 = [{1, 2}, {3, 4}, {5, 6, 7, 8}]
    sys2 = [{1, 3}, {2, 4}, {5, 6, 7, 8}]
    return ScoreHolder.apply(evaluates(gold1, sys1), intern, evaluates(gold2, sys2))


def weight_2() -> ScoreHolder[bool]:
    def intern(x: float, y: float) -> bool:
        return isclose(x, y)

    gold = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys1 = [{1, 2, 3, 4, 6}, {5, 7, 8, 9, 10}, {11, 12, 13}, {14, 15, 16}]
    sys2 = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 14}, {13, 15, 16}]
    return ScoreHolder.apply(evaluates(gold, sys1), intern, evaluates(gold, sys2))


def trivia_1(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
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
        # return EasyFail.has_passed_test(not isclose(x, 1))

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
    def intern(x: float, y: float) -> bool:
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

def metric_4b_ultra(a: Partition, b: Partition, c: Partition) -> ScoreHolder[bool]:
    """
    evaluates whether the (similarity) triangle inequality is respected
    max(s(a,b),s(b,c)) <= s(a,c)
    """

    def intern(x: float, y: float) -> bool:
        return isclose(x, y) or x < y

    return ScoreHolder.apply(ScoreHolder.apply(evaluates(a, b),max,evaluates(b, c)), intern, evaluates(a, c))


def metric_5_indiscernable(gold: Partition, sys: Partition) -> ScoreHolder[bool]:
    def intern1(x: float, y: float):
        return isclose(x, y)

    def intern2(x: bool, y: bool):
        return x & y

    a = evaluates(gold, gold)
    b = evaluates(sys, sys)
    c = evaluates(gold, sys)

    d = ScoreHolder.apply(a, intern1, b)
    e = ScoreHolder.apply(a, intern1, c)
    f = ScoreHolder.apply(b, intern1, c)

    if gold == sys:
        return ScoreHolder.apply(
            ScoreHolder.apply(d, intern2, e),
            intern2,
            f
        )
    else:
        return ScoreHolder.apply(
            ScoreHolder.apply(d, intern2, e),
            intern2,
            f
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
        size: int = 20
) -> Iterator[List[Partition], ScoreHolder[T]]:
    """
    Apply the given test function to partitions randomly generated with partition_generators
    """
    n_args = len(signature(test_func).parameters)
    mentions = list(range(size))
    # repeats the test 'repetitions' times
    for _ in range(repetitions):
        # if no partition_generator is given partition will follow a beta partition
        if partition_generators is None:
            partitions = [r_part(mentions) for _ in range(n_args)]
        else:
            if len(partition_generators) != n_args:
                raise Exception(f'got {len(partition_generators)} partition generator, expected {n_args}')
            partitions = [part(mentions) for part in partition_generators]
        res = test_func(*partitions)
        if res is not None:
            yield partitions, res


def fixed_gold_randomized_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder[T]]],
        it: Iterator[Partition],
        partition_generators: Optional[Tuple[Callable[[list], Partition], ...]] = None,
        repetitions: int = 100,
) -> Iterator[List[Partition], ScoreHolder[T]]:
    n_agrs = len(signature(test_func).parameters) - 1
    for gold in it:
        m = get_mentions(gold)
        # repeats the test 'repetitions' times
        for _ in range(repetitions):
            # if no partition_generator is given partition will follow a beta partition
            if partition_generators is None:
                partitions = [r_part(m) for _ in range(n_agrs)]
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
        repetitions: int = 100
) -> Iterator[List[Partition], ScoreHolder[T]]:
    return fixed_gold_randomized_test(test_func, iter_ancor(), partition_generators=partition_generators,
                                      repetitions=repetitions)


def variation_gold_test(
        test_func: Callable[[Partition, ...], Optional[ScoreHolder[T]]],
        it: Iterator[Partition],
        repetitions: int = 100
):
    n_agrs = len(signature(test_func).parameters) - 1
    for gold in it:
        # repeats the test 'repetitions' times
        for _ in range(repetitions):
            partitions = [introduce_randomness(gold) for _ in range(n_agrs)]
            res = test_func(gold, *partitions)
            yield partitions, res


def all_partitions_test(
        test_func: Callable[[Partition], Optional[ScoreHolder[T]]],
        i: int = 1,
        filtre_func: Callable[[Partition], bool] = lambda x: True,
        repetitions: int = 1
) -> Iterator[List[Partition], ScoreHolder[T]]:
    n_args = len(signature(test_func).parameters)
    for partitions in product(filter(filtre_func, all_partition_of_size(i)), repeat=n_args):
        for _ in range(repetitions):
            res = test_func(*partitions)
            if res is not None:
                yield list(partitions), res


mod = 3
def printmod():
    print(mod)

@dataclass
class ExpSetup:
    active: bool = False
    repetitions: int = 1
    start: int = 1
    end: int = 1


class ExpRunner:
    distributions = [r_part]  # , partial(beta_partition, a=1, b=100)]

    def __init__(
            self,
            test_func: Callable[[Partition, ...], ScoreHolder[T]],
            descr_str: str,
            agg: Tuple[Callable[[Iterator[T]], U], Callable[[Iterator[U]], V]] = (simple_and_acc, simple_and_acc),
            *,
            systematic: ExpSetup = ExpSetup(False, 1, 6, 7),
            on_corpus: ExpSetup = ExpSetup(False, 1),
            randomize: ExpSetup = ExpSetup(False, 1, 6, 7),
            variation: ExpSetup = ExpSetup(False, 30, 6, 7)
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
            try :
                yield i, self.agg[0](all_partitions_test(self.test_func, i=i, repetitions=self.systematic.repetitions))
            except :
                pass


    def intern2(self) -> Iterator[Tuple[int, T]]:
        for dists in product(ExpRunner.distributions, repeat=self.n_args):
            for i in range(self.randomize.start, self.randomize.end):
                yield dists, self.agg[0](randomized_test(
                    self.test_func,
                    partition_generators=dists,
                    repetitions=self.randomize.repetitions,
                    size= i
                    )
                )

    def intern3(self) -> Iterator[Tuple[int, T]]:
        for dists in product(ExpRunner.distributions, repeat=self.n_args - 1):
            yield dists, self.agg[0](
                ancor_gold_randomized_test(
                    self.test_func,
                    partition_generators=dists,
                    repetitions=self.on_corpus.repetitions
                )
            )

    # def intern4(self) -> Iterator[Tuple[int, T]]:
    #     for i in range(self.start, self.end):
    #         yield i, self.agg(variation_gold_test(self.test_func,
    #         it=all_partition_of_size(i), repetitions=self.repetitions))

    def f(self) -> Iterator[T]:
        if self.n_args != 0:
            if self.systematic.active:
                try:
                    yield self.agg[1](self.intern1())
                except:
                    pass
            if self.on_corpus.active:
                yield self.agg[1](self.intern3())
            # if self.variation:
            #     yield self.agg2(self.intern4())
        if self.randomize.active:
            yield self.agg[1](self.intern2())

    def g1(self) -> None:
        import csv
        fig = plt.plot()
        Y = {}
        E = {}
        C = {}
        X = []
        L = []
        for a in self.f():
            for k, (count, avg, std) in a.items():
                X.append(k)
                if not L:
                    L = list(avg.dic.keys())
                for l, i, j in zip(L, avg.dic.values(), std.dic.values()):
                    Y[l] = Y.setdefault(l, []) + [i[0]]
                    E[l] = E.setdefault(l, []) + [j[0]]
                    C[l] = C.setdefault(l, []) + [count]
        with open('data_brut3.csv', 'a', newline='') as csv_file:
            wr = csv.writer(csv_file, delimiter=';', quotechar='"')
            for c, y, e, l in zip(C.values(), Y.values(), E.values(), L):
                x, y, e, c = zip(*sorted(zip(X, y, e, c)))
                for key, avg, std, count in zip(x, y, e, c):
                    wr.writerow([mod, l, *key, "{:.2f}".format(avg), "{:.2f}".format(std), count])
                    #print([mod, l, *key, "{:.2f}".format(avg), "{:.2f}".format(std), count])

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
        plt.xlim(0, 9)
        plt.legend()
        plt.show()

    def h(self):
        map(to_tuple, self.intern1())


ALL_TESTS = {
    sing_1_a1: ExpRunner(sing_1_a1, 'a1', systematic=ExpSetup(True, 1, 1, 7)),
    sing_1_a2: ExpRunner(sing_1_a2, 'a2', systematic=ExpSetup(True, 1, 1, 7)),
    sing_1_b: ExpRunner(sing_1_b, 'a3', systematic=ExpSetup(True, 1, 1, 7)),
    biais_1: ExpRunner(biais_1, 'b1', systematic=ExpSetup(True, 1, 1, 7)),
    biais_2: ExpRunner(biais_2, 'b2', systematic=ExpSetup(True, 1, 1, 7)),
    sing_2: ExpRunner(lambda x: sing_2(x, modifications=mod), 'c',
                      agg=(series_micro_acc1, series_micro_acc2),
                      #systematic=ExpSetup(True, 30, 9, 10)
                      randomize=ExpSetup(True, 1_000_000, 20, 21)),
    weight_1: ExpRunner(weight_1, 'd1', systematic=ExpSetup(True, 1, 1, 7)),
    weight_2: ExpRunner(weight_2, 'd2', systematic=ExpSetup(True, 1, 1, 7)),
    identity_test: ExpRunner(identity_test, 'e1 | identity', randomize=ExpSetup(True, 100, 1, 7)),
    non_identity_test: ExpRunner(non_identity_test, 'e2 | non_identity', randomize=ExpSetup(True, 100, 1, 7)),
    trivia_1: ExpRunner(trivia_1, 'f', agg=(simple_or_acc, simple_or_acc), randomize=ExpSetup(True, 100, 1, 7)),
    metric_1_symetry_test: ExpRunner(metric_1_symetry_test, 'metrique 1',
                                     agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_2_non_negativity_test: ExpRunner(metric_2_non_negativity_test, 'metrique 2',
                                            agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_3: ExpRunner(metric_3, 'metrique 3',
                        agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_4b_ultra: ExpRunner(metric_4b_ultra, 'metrique 4b',
                                      agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 5)),
    metric_4_triangle_test: ExpRunner(metric_4_triangle_test, 'metrique 4',
                                      agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 5)),
    metric_5_indiscernable: ExpRunner(metric_5_indiscernable, 'metrique 5',
                                      agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_6: ExpRunner(metric_6, 'metrique 6',
                        agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_7: ExpRunner(metric_7, 'metrique 7',
                        agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    metric_8: ExpRunner(metric_8, 'metrique 8',
                        agg=(list_and_acc1, list_and_acc2), systematic=ExpSetup(True, 1, 1, 6)),
    singleton_test: ExpRunner(singleton_test, 'singleton',
                              agg=(macro_avg_acc, macro_avg_acc), randomize=ExpSetup(True, 100, 1, 7)),
    entity_test: ExpRunner(entity_test, 'entity',
                           agg=(macro_avg_acc, macro_avg_acc), randomize=ExpSetup(True, 100, 1, 7)),
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
