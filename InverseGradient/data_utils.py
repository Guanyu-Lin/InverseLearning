import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from copy import deepcopy
import random
import torch.utils.data as data


def load_all(dataset, data_path):
	Thre = 2
	train_rating = data_path + '{}.train.rating'.format(dataset)
	valid_rating = data_path + '{}.valid.rating'.format(dataset)
	test_negative = data_path + '{}.test.negative'.format(dataset)

	################# load training data #################	
	train_data = pd.read_csv(
		train_rating, 
		sep='\t', header=None, names=['user', 'item', 'rating'], 
		usecols=[0, 1, 2], dtype = {0 : np.float, 1 : np.float, 2 : np.int})
	train_data = train_data.dropna(axis = 0, how = 'any').astype(int)
	train_data = train_data.dropna(axis = 1, how = 'any').astype(int)

	# import pdb
	# pdb.set_trace()
	if dataset == "adressa":
		user_num = 212231
		item_num = 6596
	elif dataset == "kuaishou":
		user_num = 37692
		item_num = 131690
	elif dataset == "amazon_book":
		user_num = 622558 + 1
		item_num = 596401 + 1
	else:
		user_num = train_data['user'].max() + 1
		item_num = train_data['item'].max() + 1
	print("user, item num")
	print(user_num, item_num)
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	train_data_list = []
	train_data_noisy = []

	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
		if dataset!= 'kuaishou':
			if x[2] > Thre:
				label = 1
			else:
				label = 0
		else:
			label = x[2]
		train_data_list.append([x[0], x[1], label])
		train_data_noisy.append(x[2])

	################# load validation data #################
	valid_data = pd.read_csv(
		valid_rating, 
		sep='\t', header=None, names=['user', 'item', 'rating'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	valid_data = valid_data.values.tolist()
	valid_data_list = []

	for x in valid_data:
		if dataset != 'kuaishou':
			if x[2] > Thre:
				label = 1
			else:
				label = 0
		else:
			label = x[2]
		valid_data_list.append([x[0], x[1], label])
	
	user_pos = {}
	for x in train_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]
	for x in valid_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]


	################# load testing data #################
	test_data_pos = {}

	test_data = pd.read_csv(
		test_negative, 
		sep='\t', header=None, names=['user', 'item', 'rating'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	test_data = test_data.values.tolist()
	test_data_list = []

	for x in test_data:
		if dataset != 'kuaishou':
			if x[2] > Thre:
				label = 1
			else:
				label = 0
		else:
			label = x[2]

		u = x[0]
		i = x[1]
		# if label == 1:
		if u in test_data_pos:
			test_data_pos[u].append([i, label])
		else:
			test_data_pos[u] = [[i, label]]
		test_data_list.append([x[0], x[1], label])
	pop_keys = []
	for u in test_data_pos:
		label = test_data_pos[u][0][1]
		flag = 0
		for i in test_data_pos[u]:
			if (i[1] != label):
				flag = 1
				break
		if flag == 0:
			pop_keys.append(u)
			# del test_data_pos[u]
	[test_data_pos.pop(u) for u in pop_keys]
		# if (test_data_pos)
	# user_pos = {}
	# for x in train_data_list:
	# 	if x[0] in user_pos:
	# 		user_pos[x[0]].append(x[1])
	# 	else:
	# 		user_pos[x[0]] = [x[1]]
	# for x in valid_data_list:
	# 	if x[0] in user_pos:
	# 		user_pos[x[0]].append(x[1])
	# 	else:
	# 		user_pos[x[0]] = [x[1]]

	# test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	# test_data_pos = {}
	# with open(test_negative, 'r') as fd:
	# 	line = fd.readline()
	# 	while line != None and line != '':
	# 		arr = line.split('\t')
	# 		if dataset == "adressa":
	# 			u = eval(arr[0])[0]
	# 			i = eval(arr[0])[1]
	# 		else:
	# 			u = int(arr[0])
	# 			i = int(arr[1])
	# 		if u in test_data_pos:
	# 			test_data_pos[u].append(i)
	# 		else:
	# 			test_data_pos[u] = [i]
	# 		test_mat[u, i] = 1.0
	# 		line = fd.readline()


	return train_data_list, valid_data_list, test_data_list, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy

class NCFData(data.Dataset):  # train 0, validate 1
	def __init__(self, features,
				num_item, train_mat=None, num_vg=0, is_training=0):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		# self.features_ng = features_ng
		# self.features_vg = features_vg

		# if is_training == 0:
		# 	self.vg_or_not = vg_or_not
		# else:
		self.vague_or_not = [0 for _ in range(len(features))]
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_vg = num_vg
		self.is_training = is_training
		self.labels = [x[2] for x in ((features))]

	def ng_sample(self):
		assert self.is_training  != 2, 'no need to sampling when testing'

		self.features_vg = []
		# self.labels_true = []
		for x in self.features:
			u = x[0]
			# labels_true.append(x[2])
			# count += 1
			# if ()
			for t in range(self.num_vg):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_vg.append([u, j])

		# labels_true = [x[2] for x in (self.features)]
		self.labels_vg = [0 for _ in range(len(self.features_vg))]

		self.vague_or_not_fill = self.vague_or_not + [1 for _ in range(len(self.features_vg))]
		self.sample_vague = [1 for _ in range(len(self.features_vg))]
		self.features_fill = self.features + self.features_vg
		assert len(self.vague_or_not_fill) == len(self.features_fill)
		# self.labels_fill = labels_true + labels_vg

	def __len__(self):
		# return (self.num_vg + 1) * len(self.labels)
		return len(self.labels)

	def __getitem__(self, idx):
		label_len = len(self.features)

		test_user = self.features[idx % label_len][0]
		test_item = self.features[idx % label_len][1]
		test_label = self.features[idx % label_len][2]

		if self.is_training  != 0:
			return test_user, test_item, test_label
			
		features = self.features_vg
		labels = self.labels_vg
		vg_len = len(features)
		user = features[idx % vg_len][0]
		item = features[idx % vg_len][1]
		label = labels[idx % vg_len]
		return user, item, label, 1.0, 0.0, test_user, test_item, test_label
		
