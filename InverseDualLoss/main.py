import os
import time
import argparse
import numpy as np
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils
from loss import loss_function, loss_function_true

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
	default=0.01, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
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
	help="pretrain or not")
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

train_data, valid_data, test_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, 0)
valid_dataset = data_utils.NCFData(
		valid_data, item_num, train_mat, 0, 1)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, 2)
train_loader = data.DataLoader(train_dataset,
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


if args.isPretrain:
	print("loading pretrain")
	model = torch.load('{}{}LGYBase_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
	model.cuda()
else:
	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
							args.dropout, args.model, GMF_model, MLP_model)
	model.cuda()
BCE_loss = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
# if True:
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(model, valid_loader, best_loss, count, epoch, best_epoch):
	
	model.eval()
	epoch_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	for user, item, label, noisy_or_not, true_or_not, pos_lab, neg_lab in valid_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		noisy_or_not = noisy_or_not.cuda()
		true_or_not = true_or_not.cuda()
		pos_lab = pos_lab.float().cuda()
		neg_lab = neg_lab.float().cuda()

		prediction = model(user, item)
		loss = loss_function_true(prediction, label, noisy_or_not, true_or_not, pos_lab, neg_lab)
		epoch_loss += loss.detach()
	print("################### EVAL ######################")
	print("Eval loss:{}".format(epoch_loss))
	if epoch_loss < best_loss:
	# if True:
		best_loss = epoch_loss
		best_epoch = epoch
		if args.out:
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			torch.save(model, '{}{}{}_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
	return best_loss, best_epoch

########################### Test #####################################
def test(model, test_data, user_pos, best_epoch):
	top_k = args.top_k
	model.eval()
	valid_res = evaluate.test_all_users(model, 4096, item_num, test_data, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print(
		"eval valid at epoch {0}: {1}".format(
			best_epoch,
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
best_epoch = 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).

	start_time = time.time()
	train_loader.dataset.ng_sample()
	epoch_loss = 0
	for user, item, label, vague_or_not, true_or_not, pos_lab, neg_lab in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		vague_or_not = vague_or_not.cuda()
		true_or_not = true_or_not.cuda()
		pos_lab = pos_lab.float().cuda()
		neg_lab = neg_lab.float().cuda()

		# print(label.float().size())
		# model_dict = model.state_dict()
		# import pdb
		# pdb.set_trace()

		model.zero_grad()
		prediction = model(user, item)
		# print(vague_or_not)
		# print(true_or_not)
		# fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

		loss = loss_function(prediction, label, vague_or_not, true_or_not, pos_lab, neg_lab, epoch)
		# grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
		# adapted_params = OrderedDict()
		# for (key, val), grad in zip(model.named_parameters(), grads):
		# 	# Also we only need single update of inner gradient update
		# 	if (grad != None):
		# 		task_lr = args.lr * 10
		# 		adapted_params[key] = val - task_lr * grad
		# 	else:
		# 		adapted_params[key] = val
		# model.load_state_dict(adapted_params)
		epoch_loss += loss.detach()
		loss.backward()
		optimizer.step()
		# if (epoch == 0):
		# 	print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
		# if count % args.eval_freq == 0 and count != 0:
			
		count += 1

	print("epoch: {}, iter: {}, loss:{}".format(epoch, count, epoch_loss))
	best_loss = eval(model, valid_loader, best_loss, count, epoch, best_epoch)
	# test_model = torch.load('{}{}{}_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
	# test_model.cuda()
	# test(test_model, test_loader, user_pos)
	model.train()
	# print("epoch: {}, iter: {}, loss:{}".format(epoch, count, epoch_loss))
print("############################## Training End. ##############################")
test_model = torch.load('{}{}{}_{}-{}.pth'.format(model_path, args.model, args.proc, args.drop_rate, args.num_gradual))
test_model.cuda()
test(test_model, test_loader, user_pos, best_epoch)
