from __future__ import annotations

# std lib
from typing import Callable, Tuple, Union, Iterator, Optional, Any, TypeVar, List
from inspect import signature
from os import listdir
from itertools import product
from math import isclose
from functools import partial, reduce

# scorch lib
from scorch.main import clusters_from_json

# this lib
from partition_utils import Partition, singleton_partition, entity_partition, beta_partition, get_mentions, \
    all_partition_of_size, is_regular, contains_singleton
from utils import ScoreHolder, evaluates, to_tuple, simple_and_acc, simple_or_acc,\
    list_and_acc1, list_and_acc2, list_or_acc1, list_or_acc2, micro_avg_acc1, micro_avg_acc2, \
    micro_avg_std_acc1, micro_avg_std_acc2, macro_avg_acc, macro_avg_std_acc1, macro_avg_std_acc2

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


def all_partitions_test(
        test_func: Callable[[Partition], Optional[ScoreHolder[T]]],
        i: int = 1,
        filtre_func: Callable[[Partition], bool] = lambda x : True
) -> Iterator[Tuple[Partition], ScoreHolder[T]]:
    n_args = len(signature(test_func).parameters)
    for partitions in product(filter(filtre_func,all_partition_of_size(i)), repeat=n_args):
        res = test_func(*partitions)
        if res is not None:
            yield partitions, res


class tmp_class:
    distributions = [partial(beta_partition, a=1, b=1)]#, partial(beta_partition, a=1, b=100)]

    def __init__(
            self,
            test_func: Callable[[Partition, ...], ScoreHolder[T]],
            descr_str: str,
            on_corpus: bool = False,
            randomize: bool = False,
            repetitions: int = 100,
            std: bool = False,
            start: int = 1,
            end: int = 7,
            agg : Callable[[Iterator[T]], U]= simple_and_acc,
            agg2 : Callable[[Iterator[U]], V] = simple_and_acc
    ):
        self.test_func = test_func
        self.n_args = len(signature(self.test_func).parameters)
        self.descr_str = descr_str
        self.on_corpus = on_corpus
        self.randomize = randomize
        self.repetitions = repetitions
        self.std = std
        self.start = start
        self.end = end
        self.agg = agg
        self.agg2 = agg2

    #TODO std acc in new style
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

    def intern1(self) -> Iterator[Tuple[int, T]]:
        for i in range(self.start, self.end):
            try :
                yield i, self.agg(all_partitions_test(self.test_func, i=i))
            except:
                pass

    def intern2(self) -> Iterator[Tuple[int, T]]:
        for dists in product(tmp_class.distributions, repeat=self.n_args):
            yield dists, self.agg(randomized_test(self.test_func, partition_generators=dists, repetitions=self.repetitions))

    def intern3(self) -> Iterator[Tuple[int, T]]:
        for dists in product(tmp_class.distributions, repeat=self.n_args-1):
            yield dists, self.agg(
                ancor_gold_randomized_test(
                    self.test_func,
                    partition_generators=dists,
                    repetitions=self.repetitions if self.n_args != 1 else 1
                )
            )

    def f(self) -> Iterator[T]:
        if self.n_args != 0:
            try :
                yield self.agg2(self.intern1())
            except StopIteration:
                pass
            if self.on_corpus:
                yield self.agg2(self.intern3())
        if self.randomize:
            yield self.agg2(self.intern2())

    def g2(self) -> None:
        print(self.descr_str)
        for x in self.f():
            print(x)
        print('-------------')

    def h(self):
        map(to_tuple, self.intern1())


ALL_TESTS = {
    a1_test: tmp_class(a1_test, 'a1', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    a2_test: tmp_class(a2_test, 'a2', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    a3_test: tmp_class(a3_test, 'a3', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    b1_test: tmp_class(b1_test, 'b1', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    b2_test: tmp_class(b2_test, 'b2', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    d1_test: tmp_class(d1_test, 'd1', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    d2_test: tmp_class(d2_test, 'd2', repetitions=1, agg= simple_and_acc, agg2=simple_and_acc),
    identity_test: tmp_class(identity_test, 'e1 | identity', repetitions=100, agg= simple_and_acc, agg2=simple_and_acc),# agg=list_and_acc, agg2=list_and_acc_acc)
    non_identity_test: tmp_class(non_identity_test, 'e2 | non_identity', repetitions=100, start=2, end=4, agg= simple_and_acc, agg2=simple_and_acc),
    f_test: tmp_class(f_test, 'f', repetitions=100, agg=simple_or_acc, agg2=simple_or_acc),
    metric_1_symetry_test: tmp_class(metric_1_symetry_test, 'metrique 1', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_2_non_negativity_test: tmp_class(metric_2_non_negativity_test, 'metrique 2', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_3: tmp_class(metric_3, 'metrique 3', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_4_triangle_test: tmp_class(metric_4_triangle_test, 'metrique 4', end=6, repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_5_indiscernable: tmp_class(metric_5_indiscernable, 'metrique 5', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_6: tmp_class(metric_6, 'metrique 6', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_7: tmp_class(metric_7, 'metrique 7', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    metric_8: tmp_class(metric_8, 'metrique 8', repetitions=100, agg= list_and_acc1, agg2=list_and_acc2),
    singleton_test: tmp_class(singleton_test, 'singleton', repetitions=100, agg= macro_avg_acc, agg2=macro_avg_acc, randomize=True),
    entity_test: tmp_class(entity_test, 'entity', repetitions=100, agg= macro_avg_std_acc1, agg2=macro_avg_std_acc2, randomize=True),
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
