from typing import Iterator

from property_tests import ALL_TESTS, symetry_test, singleton_test, entity_test, identity_test, triangle_test
from partition_utils import beta_partition
from functools import partial
from itertools import product

x = [singleton_test, entity_test, identity_test, symetry_test, triangle_test]
for k, v in ALL_TESTS.items():
    if k in x:
        v.g2()
