import math
from functools import partial, reduce
import numpy as np
from typing import Iterator
from partition_utils import Partition

# def f(n,r):
# 	if r > n : return 0
# 	if r == 1 : return 1
# 	res = 0
# 	for p in range(1, n) :
# 		#print(p, n)
# 		x = math.comb(n -1, p) *f(p, r-1)
# 		#print(p, n, x	)
# 		res += x
# 	return res
#
# N = 10
# arr = np.array([[0]*N]*N)
# for i in range(1,N+1) :
# 	for j in range(1, N+1) :
# 		arr[i-1, j-1] = f(i, j)
#
# print(arr)

from time import time
import ast
import psycopg2
conn = psycopg2.connect("dbname=Paritions user=postgres password=root")
cur = conn.cursor()
print(cur)
#cur.execute("SELECT * FROM test;")
#print(cur.fetchone())
def timed(func, *args):
	start = time()
	res = func(*args)
	print(time() - start)
	return res







#print(list(f(3)))
#
#
# def f(n):
# 	if n == 1:
# 		yield [{1}]
# 	else:
# 		cur.execute(f"SELECT part FROM test WHERE n={n};")
# 		if cur.rowcount != 0:
# 			for partition in cur.fetchall():
# 				yield ast.literal_eval(partition[0])
# 		else:
# 			cur.execute(f"SELECT part FROM test WHERE n={n-1};")
# 			if cur.rowcount != 0:
# 				for partition in cur.fetchall():
# 					partition = ast.literal_eval(partition[0])
# 					for e, part in enumerate(partition + [set()]):
# 						res = partition[:e] + [part.union({n})] + partition[e+1:]
# 						nb_singleton, nb_cl, nb_nl, prop_cl, nb_part, format = prop(res)
# 						cur.execute(f"INSERT INTO test VALUES ('{res}', '{n}', '{nb_singleton}', '{nb_cl}', '{nb_nl}', '{prop_cl}', '{nb_part}', '{format}');")
# 						yield res
# 			else:
# 				for partition in f(n-1):
# 					for e, part in enumerate(partition + [set()]):
# 						res = partition[:e] + [part.union({n})] + partition[e+1:]
# 						nb_singleton, nb_cl, nb_nl, prop_cl, nb_part, format = prop(res)
# 						cur.execute(f"INSERT INTO test VALUES ('{res}', '{n}', '{nb_singleton}', '{nb_cl}', '{nb_nl}', '{prop_cl}', '{nb_part}', '{format}');")
# 						yield res
#
# def prop(partition):
# 	nb_mentions = 0
# 	nb_singleton = 0
# 	nb_cl = 0
# 	format = []
# 	for part in partition:
# 		size = len(part)
# 		format.append(size)
# 		nb_mentions += size
# 		if size == 1:
# 			nb_singleton+=1
# 		nb_cl += int((size * (size -1)) / 2)
# 	format = ''.join(map(str,sorted(format)))
# 	nb_l =  int((nb_mentions * (nb_mentions -1)) / 2)
# 	nb_nl = nb_l - nb_cl
# 	prop_cl = nb_cl / nb_l
# 	return nb_singleton, nb_cl, nb_nl, prop_cl, len(partition), format
#
# def timed(func, *args):
# 	start = time()
# 	res = func(*args)
# 	print(time() - start)
# 	return res
#
# def g(n):
# 	return sum(map(lambda x: 1, f(n)))
#
#
#
#
# # try :
# # 	print(timed(g, 12))
# # 	conn.commit()
# # finally:
# # 	cur.close()
# # 	conn.close()
#
# q = { 1: [[1]] }
#
# def decompose(n):
# 	try:
# 		return q[n]
# 	except:
# 		pass
# 	result = [[n]]
# 	for i in range(1, n):
# 		a = n-i
# 		R = decompose(i)
# 		for r in R:
# 			if r[0] <= a:
# 				result.append([a] + r)
# 	q[n] = result
# 	return result
#
# def j(l):
# 	nb_singleton = 0
# 	nb_cl = 0
# 	nb_nl = 0
# 	nb_l = 0
# 	prop_l = 0
# 	nb_part = 0
# 	nb_mentions = 0
# 	for x in l:
# 		nb_part+=1
# 		if x == 1:
# 			nb_singleton+=1
# 		nb_cl += int((x * (x -1)) / 2)
# 		nb_mentions += x
# 	nb_l = int((nb_mentions * (nb_mentions -1)) / 2)
# 	nb_nl = nb_l - nb_cl
# 	prop_cl = nb_cl / nb_l
# 	cur.execute(f"INSERT INTO test2 VALUES ('{l}', '{nb_mentions}', '{nb_part}', '{nb_singleton}', '{nb_cl}', '{nb_nl}', '{prop_cl}');")
# 	return nb_singleton, nb_cl, nb_nl, prop_cl, nb_part
#
# # try:
# # 	for i in range(25,30):
# # 		for partition in decompose(i):
# # 			j(partition)
# # 	conn.commit()
# # finally:
# # 	cur.close()
# # 	conn.close()
#
