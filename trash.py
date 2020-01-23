import math
from functools import partial
import numpy as np 

def f(n,r):
	if r > n : return 0
	if r == 1 : return 1
	res = 0
	for p in range(1, n) :
		#print(p, n)
		x = math.comb(n -1, p) *f(p, r-1)
		#print(p, n, x	)
		res += x
	return res

N = 10
arr = np.array([[0]*N]*N)
for i in range(1,N+1) :
	for j in range(1, N+1) :
		arr[i-1, j-1] = f(i, j)

print(arr)