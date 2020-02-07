import unittest
from random import random
from functools import reduce

from partition_utils import entity_partition, singleton_partition, partition_to_sklearn_format, get_mentions, \
    random_partition, beta_partition


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

if __name__ == '__main__':
    unittest.main()
