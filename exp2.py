from __future__ import annotations
from property_tests import ALL_TESTS, symetry_test, identity_test, triangle_test, non_identity_test
from partition_utils import beta_partition
from functools import partial


distributions = [partial(beta_partition, a=1, b=1), partial(beta_partition, a=1, b=100)]


# def f(test_func: Callable[[Partition, ...], ScoreHolder], start=1, end=5, repetitions=100):
#     print(test_func.__name__)
#     yield all_partitions_test(test_func, start=start, end=end)
#     n_args = len(signature(test_func).parameters)
#     for dists in product(distributions, repeat=n_args):
#         yield randomized_test(test_func, partition_generators=dists, repetitions=repetitions)

x = [symetry_test, identity_test, triangle_test, non_identity_test]
for k, v in ALL_TESTS.items():
    if k in x:
        v.g()

# print(ScoreHolder.average(f(identity_test)))
# print(ScoreHolder.average(f(non_identity_test, start=2)))
# print(ScoreHolder.average(f(symetry_test)))
# print(ScoreHolder.average(f(triangle_test)))



