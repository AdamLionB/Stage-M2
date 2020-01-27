from __future__ import annotations

import sys
sys.path.insert(1, '../scorch')

from scorch.main import METRICS, clusters_from_json
import scorch.scores as scores
from scorch.scores import links_from_clusters
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import beta
from random import shuffle, random
from math import ceil
from functools import reduce
import pathlib
import os
from time import time
from copy import copy
from typing import List, Set, Callable, Iterator, Tuple, Union, Optional, Dict
from itertools import zip_longest, cycle, combinations
from collections import defaultdict

def kaapa(k, r) :
	C_k, N_k = links_from_clusters(k)
	C_r, N_r = links_from_clusters(r)
	
	tp_c = len(C_k.intersection(C_r))
	tp_n = len(N_k.intersection(N_r))
	c_k, n_k = len(C_k), len(N_k)
	c_r, n_r = len(C_r), len(N_r)

	return ((tp_c + tp_n ) / (c_k + n_k) - (( (c_k * c_r ) + (n_k * n_r) ) / (c_k + n_k)**2)) / (1 - (( (c_k * c_r ) + (n_k * n_r) ) / (c_k + n_k)**2))
	#print(tp_c, tp_n, c_k, n_k, c_r, n_r)

Partition = List[Set]

def construct_partition(mentions : List, p = 0.01) -> Partition:
	shuffle(mentions)
	partitions :  Dict[Any, Set]= {}
	tmp = defaultdict(set)
	heads = {}
	for a, b in combinations(mentions, 2) :
		if a not in heads :
			heads[a] = a
		if b not in heads :
			heads[b] = b
		if random() < p :
			heads[b] = heads[a]
			tmp[a].add(b)
	for m in mentions :
		partitions[heads[m]] = partitions.setdefault(heads[m], set()).union(tmp[m])
	partitions = [v.union([k]) for k,v in partitions.items()]
	return partitions
			
def random_partition(mentions :List, distrib : Callable[[], float]) -> Partition:
	'''
	Generates a random partitions of mentions.
	The size of the clusters composing the partitions is randomly drawn following the random
	generator distrib.
	'''
	shuffle(mentions)
	partitions = []
	while len(mentions) != 0 :
		y = ceil(distrib() * len(mentions))
		partitions.append(set(mentions[:y]))
		mentions = mentions[y:]
	return partitions

def beta_partition(mentions : List, a : float, b : float) -> Partition:
	r'''
	Generates a random partitions of mentions, which cluster sizes are randomly drawn following a 
	beta distribution of parameter a, b.
	'''
	return random_partition(mentions, beta(a,b).rvs)

def entity_partition(mentions : List) -> Partition:
	return random_partition(mentions, lambda : 1)

def singleton_partition(mentions : List) -> Partition:
	return random_partition(mentions, lambda : 1/(len(mentions)+1))

class Such_random():
	def __init__(self):
		self.seed = cycle([1,2,3,4])
	def __call__(self):
		return 1 / next(self.seed)

such_random = Such_random()
def wow_partition(mentions : List) -> Parition:
	cluster = []
	while len(mentions) != 0 :
		y = ceil(such_random() * len(mentions))
		cluster.append(set(mentions[:y]))
		mentions = mentions[y:]
	return cluster

def get_mentions(partition : Partition) -> List :
	return [mention for entity in partition for mention in entity]

SK_METRICS = {
	'ARI' : metrics.adjusted_rand_score,
	'HCV' : metrics.homogeneity_completeness_v_measure,
	'AMI' : metrics.adjusted_mutual_info_score,
	'FM'  : metrics.fowlkes_mallows_score
}
#METRICS['ARI'] = metrics.adjusted_rand_score


k = [{1}, {2}, {3}, {4}, {5, 12, 14}, {6}, {7, 9}, {8}, {10}, {11}, {13}]
r1 = [{1}, {2}, {3}, {4, 6}, {5, 12}, {7, 9, 14}, {8}, {10}, {11}, {13}]
r2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}]
r3 = [{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}]


R = [r1]# [r1, r2, r3]

class Scores():
	def __init__(self, dic :Dict[str, Tuple[float, ...]]):
		self.dic = dic
	def __add__(self, other : Optional[Key_tuple]) -> Key_tuple:
		return Scores({
			sk : tuple((x + y for x,y in zip(sv, ov)))
			for (sk, sv), ov in zip(self.dic.items(), other.dic.values())
		})
	def __truediv__(self, scalar : float) -> Key_tuple:
		return Scores({
			k : tuple((x / scalar for x in v)) 
			for k,v in self.dic.items()
		})
	def __str__(self) -> str:
		res = ''
		for k, v in self.dic.items() :
			res += f'{k}\t: {v}\n'
		return res

def to_tuple(e : Union[Any, Tuple[Any]]) -> Tuple[Any] :
	if type(e) == tuple :
		return e
	return e,

def scoress_average(scoress : Iterator[Scores]) -> Scores :
	res = next(scoress)
	count = 1
	for scores in scoress :
		res = res + scores
		count +=1
	return res / count

def partition_to_classif(partition : Partition) -> List:
	tmp = {item : n for n, cluster in enumerate(partition) for item in cluster}
	return [v for k,v in sorted(tmp.items())]
		
def get_scores(gold : Partition, sys : Partition) -> Scores:
	res = {}
	for name, metric in METRICS.items():
		res[name] = to_tuple(metric(gold, sys))
	res['conll'] = to_tuple(scores.conll2012(gold, sys))
	gold = partition_to_classif(gold)
	sys = partition_to_classif(sys)
	for name, metric in SK_METRICS.items():
		res[name] = to_tuple(metric(gold, sys))
	return Scores(res)

def K() -> Iterator[Partition]:
	for n, file in enumerate(os.listdir('../ancor/json/')):
		if n > 0 : break
		yield clusters_from_json(open('../ancor/json/'+file))

def score_partitions(K : Iterator[Partition]) -> Iterator[Scores]:
	for n, k in enumerate(K) :
		R : Iterator[Partition] = (construct_partition(get_mentions(k)) for _ in range(1))
		yield scoress_average(map(lambda r : get_scores(k,r), R))


start = time()
#print(tmp(scores_all_K(K())))
gen = scoress_average(score_partitions(K()))
print(gen)
print(time()-start)

#k = clusters_from_json(open('../ancor/json/1AG0141.json'))
#print(len(k))
#print(k)

#R = (beta_partition(mentions(k), 1, 1) for _ in range(100))

# R = (entity_partition(mentions()) for _ in range(100))
# #R = (singleton_partition(mentions(k)) for _ in range(100))

#print(k)

# def lea(keys, responses):
# 	correctLinkRatio = 0
# 	allMentions = 0
# 	for entitity in keys:
# 		entity_size = len(entitity)
# 		correctlink = 0

# 		if entity_size == 1 :




