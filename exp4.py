from property_tests import ALL_TESTS, singleton_test, entity_test

x = [singleton_test, entity_test]
for k, v in ALL_TESTS.items():
    if k in x:
        v.on_corpus = True
        v.g2()