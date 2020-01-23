import sys

sys.path.insert(1, '../scorch')

from scorch.main import METRICS, clusters_from_json
import scorch.scores as scores
from scorch.scores import links_from_clusters
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import beta
from random import shuffle
from math import ceil
import pathlib


def kaapa(k, r) :
	C_k, N_k = links_from_clusters(k)
	C_r, N_r = links_from_clusters(r)
	
	tp_c = len(C_k.intersection(C_r))
	tp_n = len(N_k.intersection(N_r))
	c_k, n_k = len(C_k), len(N_k)
	c_r, n_r = len(C_r), len(N_r)

	return ((tp_c + tp_n ) / (c_k + n_k) - (( (c_k * c_r ) + (n_k * n_r) ) / (c_k + n_k)**2)) / (1 - (( (c_k * c_r ) + (n_k * n_r) ) / (c_k + n_k)**2))
	#print(tp_c, tp_n, c_k, n_k, c_r, n_r)


def random_partition(mentions, distrib) :
	shuffle(mentions)
	cluster = []
	while len(mentions) != 0 :
		y = ceil(distrib() * len(mentions))
		cluster.append(set(mentions[:y]))
		mentions = mentions[y:]
	return cluster

def beta_partition(mentions, a, b):
	return random_partition(mentions, beta(a,b).rvs)

def mentions(partition) :
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


def partition_to_classif(partition):
	tmp = {item : n for n, cluster in enumerate(partition) for item in cluster}
	return [v for k,v in sorted(tmp.items())]
		


def all_scores(gold_clusters, sys_clusters):
	res = {}
	for name, metric in METRICS.items():
		res[name] = metric(gold_clusters, sys_clusters)
	res['conll'] = scores.conll2012(gold_clusters, sys_clusters)

	gold_clusters = partition_to_classif(gold_clusters)
	sys_clusters = partition_to_classif(sys_clusters)
	for name, metric in SK_METRICS.items():
		res[name] = metric(gold_clusters, sys_clusters)
	return res
#k = r_cluster_beta(100, 1, 1)

k = clusters_from_json(open('../ancor/json/1AG0141.json'))
#print(len(k))
#print(k)

R = (beta_partition(mentions(k), 1, 1) for _ in range(5))

for r in R :
	#print(r)
	print(all_scores(k,r)['MUC'][2])
	# for name, score in all_scores(k, r).items():
		# print(f'{name=}\t: {score}')
	print()

#print(k)

# def lea(keys, responses):
# 	correctLinkRatio = 0
# 	allMentions = 0
# 	for entitity in keys:
# 		entity_size = len(entitity)
# 		correctlink = 0

# 		if entity_size == 1 :




