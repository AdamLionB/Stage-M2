import unittest
from random import random
from functools import reduce
from itertools import combinations

from partition_utils import entity_partition, singleton_partition, partition_to_sklearn_format, get_mentions, \
    random_partition, beta_partition
from utils import ScoreHolder, Growth, to_tuple, evaluate


class test_partition_utils(unittest.TestCase):
    def test_random_partition(self):
        func = random_partition
        mentions = []
        assert func(mentions, random) == []
        mentions = list(range(42))
        assert reduce(lambda x, y: x + y, map(len, func(mentions, random))) == 42
        assert len(func(mentions, lambda: 0)) == 42
        assert len(func(mentions, lambda: 0.9999)) == 1
        mentions = ['a', 'bc', 'é']
        assert reduce(lambda x, y: x + y, map(len, func(mentions, random))) == 3

    def test_beta_partition(self):
        func = beta_partition
        mentions = []
        assert func(1, 1, mentions) == []
        mentions = list(range(42))
        assert reduce(lambda x, y: x + y, map(len, func(1, 1, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(1, 100, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(100, 100, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(100, 1, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(1, 0.5, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(0.5, 0.5, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(0.5, 1, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(0.5, 2, mentions))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(2, 0.5, mentions))) == 42
        mentions = ['a', 'bc', 'é']
        assert reduce(lambda x, y: x + y, map(len, func(2, 0.5, mentions))) == 3

    def test_entity_partition(self):
        func = entity_partition
        mentions = []
        assert func(mentions) == []
        mentions = [1]
        assert func(mentions) == [{1}]
        mentions = [1, 2]
        assert func(mentions) == [{1, 2}]
        mentions = list(range(42))
        assert len(func(mentions)) == 1
        assert len(func(mentions)[0]) == 42
        mentions = ['a']
        assert func(mentions) == [{'a'}]
        mentions = ['ab']
        assert func(mentions) == [{'ab'}]
        mentions = ['é%*^$;@àç0']
        assert func(mentions) == [{'é%*^$;@àç0'}]
        mentions = ['a', 'b']
        assert func(mentions) == [{'a', 'b'}]
        mentions = [1.2]
        assert func(mentions) == [{1.2}]

    def test_singleton_partition(self):
        func = singleton_partition
        mentions = []
        assert func(mentions) == []
        mentions = [1]
        assert func(mentions) == [{1}]
        mentions = [1, 2]
        assert len(func(mentions)) == 2
        mentions = list(range(42))
        assert len(func(mentions)) == 42
        mentions = ['a']
        assert func(mentions) == [{'a'}]
        mentions = ['ab']
        assert func(mentions) == [{'ab'}]
        mentions = ['é%*^$;@àç0']
        assert func(mentions) == [{'é%*^$;@àç0'}]
        mentions = ['a', 'b']
        assert func(mentions) == [{'a'}, {'b'}]
        mentions = [1.2]
        assert func(mentions) == [{1.2}]

    def test_partition_to_sklearn_format(self):
        func = partition_to_sklearn_format
        partition = []
        assert func(partition) == []
        partition = [{1}, {4, 3}, {2, 5, 6}]
        assert func(partition) == [0, 2, 1, 1, 2, 2]
        partition = [{'b'}, {'c', 'a'}]
        assert func(partition) == [1, 0, 1]

    def test_get_mentions(self):
        func = get_mentions
        partition = []
        assert func(partition) == []
        partition = [{1}]
        assert func(partition) == [1]
        partition = [{1}, {4, 3}, {2, 5, 6}]
        assert len(func(partition)) == 6
        partition = [{'b'}, {'c', 'a'}]
        assert len(func(partition)) == 3


class test_find_better_name(unittest.TestCase):
    def test_Scores_init(self):
        dic = {}
        assert ScoreHolder(dic).dic == {}
        dic = {'a': tuple()}
        assert ScoreHolder(dic).dic == {'a': tuple()}
        assert ScoreHolder(dic).dic == dic
        dic = {'a': (1.0, 1.2)}
        assert ScoreHolder(dic).dic == {'a': (1.0, 1.2)}
        assert ScoreHolder(dic).dic == dic

    def test_Scores_get_item(self):
        dic = {'a': tuple()}
        assert ScoreHolder(dic)['a'] == tuple()
        t = (1.0,)
        dic = {'a': t}
        assert ScoreHolder(dic)['a'] == t
        dic = {'a': (1.0,), 'b': (2.0,)}
        assert ScoreHolder(dic)['b'] == (2.0,)
        try:
            x = ScoreHolder({})['a']
            assert False
        except:
            assert True

    def test_ScoreHolder_eq(self):
        assert ScoreHolder({}) == ScoreHolder({})
        assert ScoreHolder({}) != ScoreHolder({'a': (1,)})
        assert ScoreHolder({'a': (1,)}) == ScoreHolder({'a': (1,)})
        assert ScoreHolder({'a': (1,), 'b': (1, 2)}) == ScoreHolder({'a': (1,), 'b': (1, 2)})
        assert ScoreHolder({'a': (1,), 'b': (1,)}) != ScoreHolder({'a': (1,), 'b': (1, 2)})
        assert ScoreHolder({'a': (1,), 'b': (1, 2)}) != ScoreHolder({'a': (1,), 'b': (1, 3)})
        assert ScoreHolder({'a': (1,), 'b': (1, 2)}) != ScoreHolder({'b': (1, 2), 'a': (1,)})

    def test_ScoreHolder_add(self):
        assert ScoreHolder({}) == ScoreHolder({}) + ScoreHolder({})
        assert ScoreHolder({'a': (4,), 'b': (5, 7)}) == \
               ScoreHolder({'a': (1,), 'b': (1, 2)}) + ScoreHolder({'a': (3,), 'b': (4, 5)})
        assert ScoreHolder({'a': tuple()}) == \
               ScoreHolder({'a': tuple()}) + ScoreHolder({'a': tuple()})

    def test_ScoreHolder_sub(self):
        assert ScoreHolder({}) == ScoreHolder({}) - ScoreHolder({})
        assert ScoreHolder({'a': (-2,), 'b': (-3, -3)}) == \
               ScoreHolder({'a': (1,), 'b': (1, 2)}) - ScoreHolder({'a': (3,), 'b': (4, 5)})
        assert ScoreHolder({'a': tuple()}) == \
               ScoreHolder({'a': tuple()}) - ScoreHolder({'a': tuple()})

    def test_ScoreHolder_mul(self):
        assert ScoreHolder({}) == ScoreHolder({}) * 2
        assert ScoreHolder({'a': (1,)}) == ScoreHolder({'a': (1,)}) * 1
        assert ScoreHolder({'a': (1,), 'b': (2, 3)}) == ScoreHolder({'a': (1,), 'b': (2, 3)}) * 1
        assert ScoreHolder({'a': (2,), 'b': (4, 6)}) == ScoreHolder({'a': (1,), 'b': (2, 3)}) * 2

    def test_ScoreHolder_truediv(self):
        assert ScoreHolder({}) == ScoreHolder({}) / 2
        assert ScoreHolder({'a': (1,)}) == ScoreHolder({'a': (1,)}) / 1
        assert ScoreHolder({'a': (1,), 'b': (2, 3)}) == ScoreHolder({'a': (1,), 'b': (2, 3)}) / 1
        assert ScoreHolder({'a': (1,), 'b': (2, 3)}) == ScoreHolder({'a': (2,), 'b': (4, 6)}) / 2

    def test_ScoreHolder_pow(self):
        assert ScoreHolder({}) == ScoreHolder({}) ** 2
        assert ScoreHolder({'a': (1,)}) == ScoreHolder({'a': (1,)}) ** 1
        assert ScoreHolder({'a': (1,), 'b': (2, 3)}) == ScoreHolder({'a': (1,), 'b': (2, 3)}) ** 1
        assert ScoreHolder({'a': (1,), 'b': (4, 9)}) == ScoreHolder({'a': (1,), 'b': (2, 3)}) ** 2
        assert ScoreHolder({'a': (1,), 'b': (2, 3)}) == ScoreHolder({'a': (1,), 'b': (4, 9)}) ** (1 / 2)

    def test_ScoreHolder_average(self):
        func = ScoreHolder.average
        assert ScoreHolder({}) == func((ScoreHolder({}) for _ in range(1)))
        assert ScoreHolder({}) == func((ScoreHolder({}) for _ in range(2)))
        assert ScoreHolder({'a': (1,)}) == func(iter([ScoreHolder({'a': (1,)}), ScoreHolder({'a': (1,)})]))
        assert ScoreHolder({'a': (3,)}) == func(iter([ScoreHolder({'a': (2,)}), ScoreHolder({'a': (4,)})]))
        assert ScoreHolder({'a': (3, 5)}) == func(iter([ScoreHolder({'a': (2, 4)}), ScoreHolder({'a': (4, 6)})]))
        assert ScoreHolder({'a': (3, 5), 'b': (0, 1)}) == \
               func(iter([ScoreHolder({'a': (2, 4), 'b': (-1, 2)}), ScoreHolder({'a': (4, 6), 'b': (1, 0)})]))

    def test_ScoreHolder_avg_std(self):
        func = ScoreHolder.avg_std
        assert (ScoreHolder({}), ScoreHolder({})) == func((ScoreHolder({}) for _ in range(1)))
        assert (ScoreHolder({}), ScoreHolder({})) == func((ScoreHolder({}) for _ in range(2)))
        assert (ScoreHolder({'a': (1,)}), ScoreHolder({'a': (0,)})) == \
               func(iter([ScoreHolder({'a': (1,)}), ScoreHolder({'a': (1,)})]))
        assert (ScoreHolder({'a': (3,)}), ScoreHolder({'a': (1,)})) == \
               func(iter([ScoreHolder({'a': (2,)}), ScoreHolder({'a': (4,)})]))
        assert (ScoreHolder({'a': (3, 5)}), ScoreHolder({'a': (1, 1)})) == \
               func(iter([ScoreHolder({'a': (2, 4)}), ScoreHolder({'a': (4, 6)})]))
        assert (ScoreHolder({'a': (4, 5), 'b': (0, 1)}), ScoreHolder({'a': (2, 1), 'b': (1, 1)})) == \
               func(iter([ScoreHolder({'a': (2, 4), 'b': (-1, 2)}), ScoreHolder({'a': (6, 6), 'b': (1, 0)})]))
        assert (ScoreHolder({'a': (1,)}), ScoreHolder({'a': (0,)})) == \
               func(iter([ScoreHolder({'a': (1,)})]))

    def test_ScoreHolder_compare(self):
        assert ScoreHolder({}) == ScoreHolder({}).compare(ScoreHolder({}))
        assert ScoreHolder({'a': (Growth.STRICT_INCR,)}) == ScoreHolder({'a': (1,)}).compare(ScoreHolder({'a': (2,)}))
        assert ScoreHolder({'a': (Growth.STRICT_INCR, Growth.CONST, Growth.STRICT_DECR)}) == \
               ScoreHolder({'a': (1, 3, 5)}).compare(ScoreHolder({'a': (2, 3, 4)}))
        assert ScoreHolder({'a': (Growth.STRICT_INCR, Growth.CONST), 'b': (Growth.STRICT_DECR,)}) == \
               ScoreHolder({'a': (1, 3), 'b': (4,)}).compare(ScoreHolder({'a': (2, 3), 'b': (2,)}))

    def test_ScoreHolder_compare_t(self):
        assert ScoreHolder({}) == ScoreHolder({}).compare_t(tuple())
        assert ScoreHolder({'a': (Growth.STRICT_INCR,)}) == ScoreHolder({'a': (1,)}).compare_t((2,))
        assert ScoreHolder({'a': (Growth.STRICT_INCR, Growth.STRICT_DECR)}) == \
               ScoreHolder({'a': (1, 1)}).compare_t((2, 0))
        assert ScoreHolder({'a': (Growth.STRICT_INCR, Growth.STRICT_DECR), 'b': (Growth.STRICT_DECR, Growth.CONST)}) == \
               ScoreHolder({'a': (1, 1), 'b': (3, 0)}).compare_t((2, 0))

    def test_Growth(self):
        l = [Growth.STRICT_INCR, Growth.INCR, Growth.CONST, Growth.DECR, Growth.STRICT_DECR, Growth.NON_MONOTONIC]
        for x, y in combinations(l, 2):
            assert x != y

    def test_Growth_add(self):
        l = [Growth.STRICT_INCR, Growth.INCR, Growth.CONST, Growth.DECR, Growth.STRICT_DECR, Growth.NON_MONOTONIC]
        for i in l:
            assert i + i == i
            assert i + Growth.NON_MONOTONIC == Growth.NON_MONOTONIC
        assert Growth.CONST + Growth.INCR == Growth.INCR + Growth.CONST == Growth.INCR
        assert Growth.CONST + Growth.STRICT_INCR == Growth.STRICT_INCR + Growth.CONST == Growth.INCR
        assert Growth.CONST + Growth.DECR == Growth.DECR + Growth.CONST == Growth.DECR
        assert Growth.CONST + Growth.STRICT_DECR == Growth.STRICT_DECR + Growth.CONST == Growth.DECR
        assert Growth.INCR + Growth.STRICT_INCR == Growth.STRICT_INCR + Growth.INCR == Growth.INCR
        assert Growth.INCR + Growth.DECR == Growth.DECR + Growth.INCR == Growth.NON_MONOTONIC
        assert Growth.INCR + Growth.STRICT_DECR == Growth.STRICT_DECR + Growth.INCR == Growth.NON_MONOTONIC
        assert Growth.DECR + Growth.STRICT_INCR == Growth.STRICT_INCR + Growth.DECR == Growth.NON_MONOTONIC
        assert Growth.DECR + Growth.STRICT_DECR == Growth.STRICT_DECR + Growth.DECR == Growth.DECR
        assert Growth.STRICT_INCR + Growth.STRICT_DECR == Growth.STRICT_DECR + Growth.STRICT_INCR == \
               Growth.NON_MONOTONIC

    def test_Growth_truediv(self):
        l = [Growth.STRICT_INCR, Growth.INCR, Growth.CONST, Growth.DECR, Growth.STRICT_DECR, Growth.NON_MONOTONIC]
        for i in l:
            assert i / 3.14 == i
            assert i / i == i

    def test_Growth_compare(self):
        func = Growth.compare
        assert func(2, 3) == Growth.STRICT_INCR
        assert func(4, 3) == Growth.STRICT_DECR
        assert func(4, 4) == Growth.CONST

    def test_to_tuple(self):
        func = to_tuple
        assert (1,) == func(1)
        assert (1,) == func((1,))
        assert (1, 2) == func((1, 2))

if __name__ == '__main__':
    unittest.main()
