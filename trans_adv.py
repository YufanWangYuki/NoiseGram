import torch
torch.cuda.empty_cache()
import random
import time
import os
import argparse
import sys
import numpy as np

from utils.misc import set_global_seeds, save_config, validate_config, check_device
from utils.dataset import Dataset
from models.Seq2seq import Seq2seq
from trainer.trasner_adv7 import Trainer


def load_arguments(parser):

	""" Seq2seq model """

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, required=True, help='train tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')

	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_mode', type=str, default='null', help='loading mode resume|restart|null')

	# train
	parser.add_argument('--max_src_len', type=int, default=32, help='maximum src sequence length')
	parser.add_argument('--max_tgt_len', type=int, default=32, help='maximum tgt sequence length')
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')
	parser.add_argument('--gpu_id', type=int, default=0, help='only used for memory reservation')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')

	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--minibatch_split', type=int, default=1, help='split the batch to avoid OOM')

	parser.add_argument('--lr_peak', type=float, default=0.00001, help='learning rate')
	parser.add_argument('--lr_init', type=float, default=0.0005, help='learning rate init')
	parser.add_argument('--lr_warmup_steps', type=int, default=12000, help='lr warmup steps')
	parser.add_argument('--max_grad_norm', type=float, default=1.0,
		help='optimiser gradient norm clipping: max grad norm')

	# save and print
	parser.add_argument('--grab_memory', type=str, default='True', help='grab full GPU memory')
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')
	parser.add_argument('--max_count_no_improve', type=int, default=2,
		help='if meet max, operate roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=2,
		help='if meet max, reduce learning rate')
	parser.add_argument('--keep_num', type=int, default=1,
		help='number of models to keep')

	# noise
	parser.add_argument('--ntype', type=str, default='Gaussian',
		help='noise type')
	parser.add_argument('--nway', type=str, default='mul',
		help='noise add way: mul or add')
	parser.add_argument('--mean', type=float, default=1.0,
		help='noise mean')
	parser.add_argument('--weight', type=float, default=0.0,
		help='noise weight')
	parser.add_argument('--word_keep', type=float, default=1.0,
		help='word keep')
	parser.add_argument('--replace_map', type=list, default=None,
		help='replace map')

	return parser


def main():

	# load config
	parser = argparse.ArgumentParser(description='T5 Fintuning for EN-DE translation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# loading old models
	if config['load']:
		print('loading {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					load_mode=config['load_mode'],
					batch_size=config['batch_size'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					lr_peak=config['lr_peak'],
					lr_init=config['lr_init'],
					lr_warmup_steps=config['lr_warmup_steps'],
					use_gpu=config['use_gpu'],
					gpu_id=config['gpu_id'],
					max_grad_norm=config['max_grad_norm'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					keep_num=config['keep_num'],
					minibatch_split=config['minibatch_split'],
					noise_type=config['ntype'],
					weight=config['weight'],
					mean=config['mean'],
					word_keep=config['word_keep'],
					noise_way=config['nway']
					)

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	train_set = Dataset(train_path_src, train_path_tgt,
		max_src_len=config['max_src_len'],
		max_tgt_len=config['max_tgt_len'],
		batch_size=config['batch_size'],
		use_gpu=config['use_gpu'],
		logger=t.logger)

	# load dev set
	if config['dev_path_src'] and config['dev_path_tgt']:
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_set = Dataset(dev_path_src, dev_path_tgt,
			max_src_len=config['max_src_len'],
			max_tgt_len=config['max_tgt_len'],
			batch_size=config['batch_size'],
			use_gpu=config['use_gpu'],
			logger=t.logger)
	else:
		dev_set = None

	# construct model
	seq2seq = Seq2seq()

	t.logger.info("total #parameters:{}".format(sum(p.numel() for p in
		seq2seq.parameters() if p.requires_grad)))

	device = check_device(config['use_gpu'])
	t.logger.info('device: {}'.format(device))
	seq2seq = seq2seq.to(device=device)

	t.logger.info('noise: %s weight: %s mean: %s word_keep: %s noise_way: %s '.format(config['ntype'],config['weight'],config['mean'],config['word_keep'],config['nway']))

	# run training
	seq2seq = t.train(train_set, seq2seq, num_epochs=config['num_epochs'],
		dev_set=dev_set, grab_memory=config['grab_memory'])


if __name__ == '__main__':
	main()
