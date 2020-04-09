from property_tests import ALL_TESTS, c_test
from utils import SK_METRICS

for _ in range(len(SK_METRICS.keys())):
    SK_METRICS.popitem()

x = [c_test]

for k,v in ALL_TESTS.items():
    if k in x:
        v.g1()