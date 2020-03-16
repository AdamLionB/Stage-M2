from typing import Iterator

from property_tests import ALL_TESTS, metric_1_symetry_test, singleton_test, entity_test, identity_test, metric_4_triangle_test
from partition_utils import beta_partition
from functools import partial
from itertools import product

x = [singleton_test, entity_test, identity_test, metric_1_symetry_test, metric_4_triangle_test]
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()
