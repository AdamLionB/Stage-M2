from typing import Callable
from time import time
class Timer:
    def __init__(self):
        self.dic = {}

    def timed(self, func: Callable) -> Callable:
        self.dic[func] = 0

        def intern(*args, **kwargs):
            start = time()
            res = func(*args, **kwargs)
            self.dic[func] += time() - start
            return res
        return intern

    def __repr__(self):
        res = ''
        for k, v in self.dic.items():
            res += f'{k.__name__}\t\t: {v}\n'
        return res
