from __future__ import annotations
from property_tests import ALL_TESTS, metric_1_symetry_test, metric_2_non_negativity_test, metric_3, \
    metric_4_triangle_test, metric_5_indiscernable, metric_6, metric_7, metric_8, all_partitions_test
from utils import SK_METRICS
from partition_utils import is_regular, contains_singleton, contains_one_entity, is_dual

for _ in range(len(SK_METRICS.keys())):
    SK_METRICS.popitem()




x = [metric_1_symetry_test, metric_2_non_negativity_test, metric_3, \
    metric_4_triangle_test, metric_5_indiscernable, metric_6, metric_7, metric_8]

print('all')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

all_partitions_test.__defaults__ = (1, is_regular)
print('no one entity, no dual')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

all_partitions_test.__defaults__ = (1, lambda x: (not contains_one_entity(x)) and (not contains_singleton(x)))
print('no one entity, no singleton')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

all_partitions_test.__defaults__ = (1, lambda x: not contains_one_entity(x))
print('no one entity')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

all_partitions_test.__defaults__ = (1, lambda x: not is_dual(x))
print('no dual')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

all_partitions_test.__defaults__ = (1, lambda x: not contains_singleton(x))
print('no singletons')
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()

