from __future__ import annotations
from testes import a_test, r_test, ancor_test, symetry_test, identity_test, kinda_triangle_test, non_identity_test
from partition_utils import beta_partition, Partition
from find_better_name import ScoreHolder
from functools import partial
from itertools import product
from inspect import signature
from typing import Callable


distributions = [partial(beta_partition, a=1, b=1), partial(beta_partition, a=1, b=100)]


def f(test_func: Callable[[Partition, ...], ScoreHolder], start=1, end=5, repetitions=100):
    print(test_func.__name__)
    yield a_test(test_func, start=start, end=end)
    n_args = len(signature(test_func).parameters)
    for dists in product(distributions, repeat=n_args):
        yield r_test(test_func, partition_generators=dists, repetitions=repetitions)

print(ScoreHolder.average(f(identity_test)))
print(ScoreHolder.average(f(non_identity_test, start=2)))
print(ScoreHolder.average(f(symetry_test)))
print(ScoreHolder.average(f(kinda_triangle_test, repetitions=200)))



# print('non_identity')
# for i in a_test(non_identity_test, start=2, end=5):
#     print(i)
#
# print('symetry')
# for i in a_test(symetry_test, end=5):
#     print(i)
#
# print('triangle')
# for i in a_test(triangle_test, end=5):
#     print(i)