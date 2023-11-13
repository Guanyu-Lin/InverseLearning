import os
import time
import argparse
import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter
from collections import OrderedDict
import pandas as pd
import model
import evaluate
import data_utils
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'amazon_book')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--sample_lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--train_ratio", 
	type=float,
	default=0.9,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--eval_freq", 
	type=int,
	default=2000,
	help="the freq of eval")
parser.add_argument('--proc', 
	type = str,
	help = 'process title',
	default = 'LGY')
parser.add_argument("--top_k", 
	type=list, 
	default=[50, 100],
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=1, 
	help="sample negative items for training")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--isPretrain", 
	default=False,
	help="pretrain model or not")
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
SEED = 42
torch.manual_seed(SEED) # cpu
torch.cuda.manual_seed(SEED) #gpu
np.random.seed(SEED) #numpy
random.seed(SEED) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)

data_path = '../data/{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
print("arguments: %s " %(args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data, test_data_user, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)
train_num = int(len(train_data) * args.train_ratio)

# construct the train and test datasets
train_dataset_train = data_utils.NCFData(
		train_data[:train_num], item_num, train_mat, args.num_ng, 0)

train_dataset_test = data_utils.NCFData(
		train_data[train_num:], item_num, train_mat, args.num_ng, 0)

valid_dataset = data_utils.NCFData(
		valid_data, item_num, train_mat, 0, 1)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, 2)
train_loader_train = data.DataLoader(train_dataset_train,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
train_loader_test = data.DataLoader(train_dataset_test,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data)))

########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre': # pre-training. Not used in our work.
	GMF_model_path = model_path + 'GMF.pth'
	MLP_model_path = model_path + 'MLP.pth'
	NeuMF_model_path = model_path + 'NeuMF.pth'
	assert os.path.exists(GMF_model_path), 'lack of GMF model'
	assert os.path.exists(MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(GMF_model_path)
	MLP_model = torch.load(MLP_model_path)
else:
	GMF_model = None
	MLP_model = None



if (args.isPretrain):
	print("loading pretrain")
	model = torch.load('{}{}{}Base_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
else :
	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model, GMF_model, MLP_model)
model.cuda()
model_forward = copy.deepcopy(model)
model_inverse = copy.deepcopy(model)
BCE_loss = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer_forward = optim.Adam(model_forward.parameters(), lr=args.lr)
optimizer_inverse = optim.Adam(model_inverse.parameters(), lr=args.lr)

# writer = SummaryWriter(model_path+'summary/') # for visualization

# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(model, valid_loader, best_loss, count, epoch, df):
	
	model.eval()
	epoch_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	for user, item, label in valid_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
	

		prediction = model(user, item)
		loss = loss_function_test(prediction, label)
		epoch_loss += loss.detach()
	print("################### EVAL ######################")
	print("Eval loss:{}".format(epoch_loss))
	if epoch_loss < best_loss:
	# if True:
		best_loss = epoch_loss
		if args.out:
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			df.to_csv('{}{}{}_{}-{}_final.csv'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
			torch.save(model, '{}{}{}_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
	return best_loss

########################### Test #####################################
def test(model, test_data, user_pos):
	top_k = args.top_k
	model.eval()
	valid_res = evaluate.test_all_users(model, 4096, item_num, test_data, test_data_user, user_pos, top_k)

	print("################### TEST ######################")
	print(
		"eval valid at epoch {0}: {1}".format(
			1,
			",".join(
				[
					"" + str(key) + ":" + str(value)
					for key, value in valid_res.items()
				]
			),
		)
	)

########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9
# loss_record = []
fp = open('{}{}{}_{}-{}_loss.csv'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual), 'w')

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).

	start_time = time.time()
	train_loader_train.dataset.ng_sample()

	for user, item, label, pos_lab, neg_lab, test_user, test_item, test_label in train_loader_train:
		test_user = test_user.cuda()
		test_item = test_item.cuda()
		test_label = test_label.float().cuda()
		prediction = model(test_user, test_item)
		loss = loss_function_test(prediction, test_label)		
		loss.backward()
		optimizer.step()

	train_loader_test.dataset.ng_sample()
	epoch_loss = 0
	# user_list = []
	# item_list = []
	flag = True
	df = None
	for user, item, label, pos_lab, neg_lab, test_user, test_item, test_label in train_loader_test:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		# vague_or_not = vague_or_not.cuda()
		# true_or_not = true_or_not.cuda()
		pos_lab = pos_lab.float().cuda()
		neg_lab = neg_lab.float().cuda()
		test_user = test_user.cuda()
		test_item = test_item.cuda()
		test_label = test_label.float().cuda()
		# print(label.float().size())
		# pdb.set_trace()
		model.zero_grad()
		model_forward.load_state_dict(model.state_dict())
		model_inverse.load_state_dict(model.state_dict())
		optimizer_forward = optim.Adam(model_forward.parameters(), lr= args.sample_lr)
		optimizer_inverse = optim.Adam(model_inverse.parameters(), lr= args.sample_lr)

		prediction_forward = model_forward(user, item)
		prediction_inverse = model_inverse(user, item)

		# print(vague_or_not)
		# print(true_or_not)
		# fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
		loss_forward, posWeight, negWeight = loss_function_train(prediction_forward, label, pos_lab, neg_lab)
		loss_inverse, _, _ = loss_function_train(prediction_inverse, label, pos_lab, neg_lab)
		loss_inverse = -loss_inverse
		loss_forward.backward()
		loss_inverse.backward()
		optimizer_forward.step()
		optimizer_inverse.step()
		optimizer_forward = optim.Adam(model_forward.parameters(), lr=args.lr)
		optimizer_inverse = optim.Adam(model_inverse.parameters(), lr=args.lr)
		
		prediction_mid = model(test_user, test_item)

		prediction_forward = model_forward(test_user, test_item)
		prediction_inverse = model_inverse(test_user, test_item)

		loss_forward = loss_function_test(prediction_forward, test_label)
		loss_inverse = loss_function_test(prediction_inverse, test_label)
		loss_mid = loss_function_test(prediction_mid, test_label)
		fp.write(str(count) +'\t' + str(loss_forward.cpu().detach().numpy()) + '\t' + str(loss_mid.cpu().detach().numpy()) + '\t' + str(loss_inverse.cpu().detach().numpy()) + '\n')

		# writer.add_scalar('loss_forward', loss_forward, count)
		# writer.add_scalar('loss_mid', loss_mid, count)
		# writer.add_scalar('loss_inverse', loss_inverse, count)
		# print(round(loss_forward, 8), round(loss_mid, 8), round(loss_inverse, 8))
		if (loss_forward < loss_inverse and loss_forward < loss_mid):
			loss_forward.backward()
			optimizer_forward.step()
			model.load_state_dict(model_forward.state_dict())
			epoch_loss += loss_forward.detach()
			# print("pos")
		elif (loss_inverse < loss_mid) :

			loss_inverse.backward()
			optimizer_inverse.step()
			model.load_state_dict(model_inverse.state_dict())
			epoch_loss += loss_inverse.detach()
			# print("neg")
		else:
			# import pdb
			# pdb.set_trace()
			loss_mid.backward()
			optimizer.step()
			# model.load_state_dict(model_inverse.state_dict())
			epoch_loss += loss_mid.detach()
			# print("cool, mid")
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		# if (epoch == 0):
		# 	print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
		# if count % args.eval_freq == 0 and count != 0:
		# loss_record.append([loss_forward, loss_mid, loss_inverse])
		count += 1
		if (flag):
			user_df = pd.DataFrame(user.cpu().detach().numpy(), columns=['user'])
			item_df = pd.DataFrame(item.cpu().detach().numpy(), columns=['item'])
			posWeight_df = pd.DataFrame(posWeight.cpu().detach().numpy(), columns=['pos'], dtype='f')
			negWeight_df = pd.DataFrame(negWeight.cpu().detach().numpy(), columns=['neg'], dtype='f')
			df = pd.concat([user_df, item_df, posWeight_df, negWeight_df], axis = 1)
			if (epoch == 0):
				df.to_csv('{}{}{}_{}-{}_epoch0.csv'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
		flag = False
	# if epoch == 0:
	# 	fp = open('posweight.txt', 'w')
	# 	# for i in train_idx:
	# 	# 	line = lines[i]
	# 	# 	user, item, rating, time = line.split('::') 
	# 	# 	user_idx = int(user)-1
	# 	# 	item_idx = int(item)-1  
	# 	# 	fp.write(user + '\t' + item + '\t' + rating + '\n')

	# 	fp.close()
	print("epoch: {}, iter: {}, loss:{}".format(epoch, count, epoch_loss))
	best_loss = eval(model, valid_loader, best_loss, count, epoch, df)
	model.train()
fp.close()
	# print("epoch: {}, iter: {}, loss:{}".format(epoch, count, epoch_loss))
print("############################## Training End. ##############################")
# loss_record_df = pd.DataFrame(loss_record.cpu().detach().numpy(), columns=['count', 'forward', 'mid', 'inverse'])
# loss_record_df.to_csv('{}{}{}_{}-{}_loss.csv'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
test_model = torch.load('{}{}{}_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
test_model.cuda()
test(test_model, test_loader, user_pos)
