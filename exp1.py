from property_tests import randomized_test, ancor_gold_randomized_test, symetry_test, singleton_test, entity_test, identity_test, distance_triangle_test
from partition_utils import beta_partition
from utils import ScoreHolder
from functools import partial
from itertools import product


distributions = [partial(beta_partition, a=1, b=1), partial(beta_partition, a=1, b=100)]
A = {1: {singleton_test: ('test singleton', True),
         entity_test: ('test entité', True),
         identity_test: ('test identité', False)},
     2: {symetry_test: ('test de symetry', False)},
     3: {distance_triangle_test: ('test id triangulaire', False)}
     }

B = {1: {singleton_test: ('test singleton', True),
         entity_test: ('test entité', True),
         identity_test: ('test identité', False)},
     2: {symetry_test: ('test de symetry', False)},
     3: {distance_triangle_test: ('test id triangulaire', False)}
     }

def g(repeat, test, std):
    for prod in product(distributions, repeat=repeat):
        print(prod)
        res = randomized_test(test, partition_generators=prod, std=std)
        print(res)
        yield res

def h(repeat, test, std):
    for prod in product(distributions, repeat=repeat):
        print(('ancor', *prod))
        if repeat == 0:
            res = ancor_gold_randomized_test(test, repetitions=1, std=std)
        else:
            res = ancor_gold_randomized_test(test, partition_generators=prod, std=std)
        print(res)
        yield res

def f():
    for repeat, tests in A.items():
        for test, (description, std) in tests.items():
            print(description)
            if std:
                list(g(repeat, test, std))
            else:
                print(ScoreHolder.average(g(repeat, test, std)))

    for repeat, tests in B.items():
        for test, (description, std) in tests.items():
            print(description)
            if std:
                list(h(repeat-1, test, std))
            else:
                print(ScoreHolder.average(h(repeat-1, test, std)))

f()
# print(r_test(symetry_test, partition_generators=(partial(beta_partition, a=1, b=1), partial(beta_partition, a=1, b=1))))
