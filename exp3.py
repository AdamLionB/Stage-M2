from property_tests import ALL_TESTS, sing_2, printmod
import property_tests
from utils import SK_METRICS
from time import time

for _ in range(len(SK_METRICS.keys())):
    SK_METRICS.popitem()



start = time()
x = [sing_2]
for i in range(1, 4):
    property_tests.mod = i
    for k,v in ALL_TESTS.items():
        if k in x:
            v.g1()
print(time() - start)