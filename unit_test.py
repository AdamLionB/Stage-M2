import unittest
from random import random
from functools import reduce
from itertools import combinations

import partition_utils
from utils import ScoreHolder, to_tuple, evaluates


class test_partition_utils(unittest.TestCase):
    def test_random_partition(self):
        func = partition_utils.random_partition
        mentions = []
        assert func(mentions, random) == []
        mentions = list(range(42))
        assert reduce(lambda x, y: x + y, map(len, func(mentions, random))) == 42
        assert len(func(mentions, lambda: 0)) == 42
        assert len(func(mentions, lambda: 0.9999)) == 1
        mentions = ['a', 'bc', 'é']
        assert reduce(lambda x, y: x + y, map(len, func(mentions, random))) == 3

    def test_beta_partition(self):
        func = partition_utils.beta_partition
        mentions = []
        assert func(mentions, 1, 1) == []
        mentions = list(range(42))
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 1, 1))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 1, 100))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 100, 100))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 100, 1))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 1, 0.5))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 0.5, 0.5))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 0.5, 1))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 0.5, 2))) == 42
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 2, 0.5))) == 42
        mentions = ['a', 'bc', 'é']
        assert reduce(lambda x, y: x + y, map(len, func(mentions, 2, 0.5))) == 3

    def test_entity_partition(self):
        func = partition_utils.entity_partition
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
        func = partition_utils.singleton_partition
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
        func = partition_utils.partition_to_sklearn_format
        partition = []
        assert func(partition) == []
        partition = [{1}, {4, 3}, {2, 5, 6}]
        assert func(partition) == [0, 2, 1, 1, 2, 2]
        partition = [{'b'}, {'c', 'a'}]
        assert func(partition) == [1, 0, 1]

    def test_get_mentions(self):
        func = partition_utils.get_mentions
        partition = []
        assert func(partition) == []
        partition = [{1}]
        assert func(partition) == [1]
        partition = [{1}, {4, 3}, {2, 5, 6}]
        assert len(func(partition)) == 6
        partition = [{'b'}, {'c', 'a'}]
        assert len(func(partition)) == 3

    def test_all_partition_of_size(self):
        func = partition_utils.all_partition_of_size
        sizes = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
        for i in range(10):
            assert sum((1 for _ in func(i+1))) == sizes[i]

    def test_all_partition_up_to_size(self):
        func = partition_utils.all_partition_up_to_size
        sizes = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
        assert sum((1 for _ in func(10) )) == sum(sizes)

    def test_is_regular(self):
        func = partition_utils.is_regular
        assert not func([{1, 2, 3, 4}])
        assert func([{1, 2, 3}, {4}])
        assert func([{1, 2}, {3, 4}])
        assert func([{1, 2}, {3}, {4}])
        assert not func([{1}, {2}, {3}, {4}])

    def test_contains_singleton(self):
        func = partition_utils.contains_singleton
        assert not func([{1, 2, 3, 4}])
        assert func([{1, 2, 3}, {4}])
        assert not func([{1, 2}, {3, 4}])
        assert func([{1, 2}, {3}, {4}])
        assert func([{1}, {2}, {3}, {4}])

    def test_contains_one_entity(self):
        func = partition_utils.contains_one_entity
        assert func([{1, 2, 3, 4}])
        assert not func([{1, 2, 3}, {4}])
        assert not func([{1, 2}, {3, 4}])
        assert not func([{1, 2}, {3}, {4}])
        assert not func([{1}, {2}, {3}, {4}])

    def test_is_dual(self):
        func = partition_utils.is_dual
        assert not func([{1, 2, 3, 4}])
        assert not func([{1, 2, 3}, {4}])
        assert not func([{1, 2}, {3, 4}])
        assert not func([{1, 2}, {3}, {4}])
        assert func([{1}, {2}, {3}, {4}])

    def test_get_partition_size(self):
        func = partition_utils.get_partition_size
        assert func([]) == 0
        assert func([{1}]) == 1
        assert func([{1, 2}]) == 2
        assert func([{1}, {2}]) == 2
        assert func([{1, 2, 3}]) == 3
        assert func([{1, 2}, {3}]) == 3
        assert func([{1}, {2}, {3}]) == 3

    def test_introduce_randomness(self):
        #TODO make statistical test ? some like 'random mutation of random partition follows a uniform distrib'
        func = partition_utils.introduce_randomness
        size_func = partition_utils.get_partition_size
        for part in partition_utils.all_partition_of_size(7):
            assert size_func(func(part)) == size_func(part)

    def test_bell(self):
        """
        Only test if this function is good enough for our purpose
        """
        func = partition_utils.bell
        b = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545, \
             10480142147, 82864869804, 682076806159, 5832742205057, 51724158235372]
        for i in range(len(b)):
            assert func(i+1) == b[i] or func(i+1) == b[i]+1

    def test_r_part(self):
        #TODO make statistical test ? some like 'random partition follows a uniform distrib'
        func = partition_utils.r_part
        size_func = partition_utils.get_partition_size
        for _ in range(877*2):
            assert size_func(func(range(7))) == 7



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

    # def test_ScoreHolder_average(self):
    #     func = ScoreHolder.average
    #     assert ScoreHolder({}) == func((ScoreHolder({}) for _ in range(1)))
    #     assert ScoreHolder({}) == func((ScoreHolder({}) for _ in range(2)))
    #     assert ScoreHolder({'a': (1,)}) == func(iter([ScoreHolder({'a': (1,)}), ScoreHolder({'a': (1,)})]))
    #     assert ScoreHolder({'a': (3,)}) == func(iter([ScoreHolder({'a': (2,)}), ScoreHolder({'a': (4,)})]))
    #     assert ScoreHolder({'a': (3, 5)}) == func(iter([ScoreHolder({'a': (2, 4)}), ScoreHolder({'a': (4, 6)})]))
    #     assert ScoreHolder({'a': (3, 5), 'b': (0, 1)}) == \
    #            func(iter([ScoreHolder({'a': (2, 4), 'b': (-1, 2)}), ScoreHolder({'a': (4, 6), 'b': (1, 0)})]))
    #
    # def test_ScoreHolder_avg_std(self):
    #     func = ScoreHolder.avg_std
    #     assert (ScoreHolder({}), ScoreHolder({})) == func((ScoreHolder({}) for _ in range(1)))
    #     assert (ScoreHolder({}), ScoreHolder({})) == func((ScoreHolder({}) for _ in range(2)))
    #     assert (ScoreHolder({'a': (1,)}), ScoreHolder({'a': (0,)})) == \
    #            func(iter([ScoreHolder({'a': (1,)}), ScoreHolder({'a': (1,)})]))
    #     assert (ScoreHolder({'a': (3,)}), ScoreHolder({'a': (1,)})) == \
    #            func(iter([ScoreHolder({'a': (2,)}), ScoreHolder({'a': (4,)})]))
    #     assert (ScoreHolder({'a': (3, 5)}), ScoreHolder({'a': (1, 1)})) == \
    #            func(iter([ScoreHolder({'a': (2, 4)}), ScoreHolder({'a': (4, 6)})]))
    #     assert (ScoreHolder({'a': (4, 5), 'b': (0, 1)}), ScoreHolder({'a': (2, 1), 'b': (1, 1)})) == \
    #            func(iter([ScoreHolder({'a': (2, 4), 'b': (-1, 2)}), ScoreHolder({'a': (6, 6), 'b': (1, 0)})]))
    #     assert (ScoreHolder({'a': (1,)}), ScoreHolder({'a': (0,)})) == \
    #            func(iter([ScoreHolder({'a': (1,)})]))

    def test_to_tuple(self):
        func = to_tuple
        assert (1,) == func(1)
        assert (1,) == func((1,))
        assert (1, 2) == func((1, 2))

if __name__ == '__main__':
    unittest.main()
