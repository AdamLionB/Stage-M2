from collections import defaultdict, Set
from random import random

def r_part(n):
  dic = defaultdict(list)
  firsts_q = [q(n, 1)]
  for i in range(n):
    r = random()
    urn = 1
    if urn > len(firsts_q):
      firsts_q.append(firsts_q[-1]+ q(n, urn))
    if r < firsts_q[urn -1]:
      dic[urn] = dic[urn] + [i]
    else:
      urn +=1
    return [Set(entity) for entity in dic.values() if not entity]