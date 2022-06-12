from itertools import count
import torch
torch.cuda.empty_cache()
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np
import torchtext
import errant
from utils.misc import get_memory_alloc, check_device, reserve_memory
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq
import string

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)

import pdb
from tqdm import tqdm

class Trainer(object):

	def __init__(self,
		expt_dir='experiment',
		load_dir=None,
		load_mode='null',
		checkpoint_every=100,
		print_every=100,
		batch_size=256,
		use_gpu=False,
		gpu_id=0,
		lr_peak=0.00001,
		lr_init=0.0005,
		lr_warmup_steps=16000,
		max_grad_norm=1.0,
		max_count_no_improve=2,
		max_count_num_rollback=2,
		keep_num=1,
		minibatch_split=1,
		noise_type='Gaussian',
		weight=0.0,
		mean=1.0,
		word_keep=1.0,
		replace_map=None,
		noise_way='mul',
		seq_length=64,
		embedding_dim=768
		):

		self.use_gpu = use_gpu
		self.gpu_id = gpu_id
		self.device = check_device(self.use_gpu)
		# self.device = torch.device("cuda:0")

		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every

		self.lr_peak = lr_peak
		self.lr_init = lr_init
		self.lr_warmup_steps = lr_warmup_steps
		if self.lr_warmup_steps == 0: assert self.lr_peak == self.lr_init

		self.max_grad_norm = max_grad_norm
		self.max_count_no_improve = max_count_no_improve
		self.max_count_num_rollback = max_count_num_rollback
		self.keep_num = keep_num
		self.max_tgt_len = 100
		self.mode = 'beam-1'

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir
		self.load_mode = load_mode
		if type(self.load_dir) != type(None) and load_mode == 'null':
			self.load_mode = 'resume' # default to resume

		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)

		self.minibatch_split = minibatch_split
		self.batch_size = batch_size
		self.minibatch_size = int(self.batch_size / self.minibatch_split) # to be changed if OOM
		self.seq_length = seq_length
		self.embedding_dim = embedding_dim

		self.noise_configs = {
			'noise':2,
			'noise_type':noise_type,
			'weight':weight,
			'mean':mean,
			'word_keep':word_keep,
			'replace_map':replace_map,
			'noise_way':noise_way
		}

		self.noise = None

		if noise_type == 'Adversarial':
			self.noise = np.ones([self.minibatch_size, seq_length, embedding_dim])
		elif noise_type == 'Gaussian-adversarial' or noise_type == 'Gaussian-adversarial-norm':
			self.noise = np.random.normal(mean, weight, [1, 1, embedding_dim])
			self.noise = torch.tensor(self.noise).to(device=self.device).expand([self.minibatch_size,seq_length,embedding_dim])
			self.noise.requires_grad = True
		elif noise_type == 'Gaussian-adversarial-diff':
			self.noise = np.random.normal(1, weight, [self.minibatch_size, seq_length, embedding_dim])
		# pdb.set_trace()
		# self.noise = torch.tensor(self.noise).to(device=self.device)
		
		self.weight = weight
		self.alpha =  1000000 # 1000000
		self.gamma = 0.5
		self.total_noise_edits = 0
		self.total_trans_edits = 0
		self.final_pred = []
		if 'norm' in noise_type:
			self.alpha = weight
		print(self.alpha)


	def _print_hyp(self, out_count, tgt_seqs, preds):

		if out_count < 3:
			outref = 'REF: {}\n'.format(tgt_seqs[0]).encode('utf-8')
			outline = 'GEN: {}\n'.format(preds[0]).encode('utf-8')
			sys.stdout.buffer.write(outref)
			sys.stdout.buffer.write(outline)
			out_count += 1
		return out_count


	def lr_scheduler(self, optimizer, step,
		init_lr=0.00001, peak_lr=0.0005, warmup_steps=16000):

		""" Learning rate warmup + decay """

		# deactivate scheduler
		if warmup_steps <= 0:
			return optimizer

		# activate scheduler
		if step <= warmup_steps:
			lr = step * 1. * (peak_lr - init_lr) / warmup_steps + init_lr
		else:
			# lr = peak_lr * ((step - warmup_steps) ** (-0.5))
			lr = peak_lr * (step ** (-0.5)) * (warmup_steps ** 0.5)

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		return optimizer


	# def _evaluate_batches(self, model, dataset):

	# 	# import pdb; pdb.set_trace()

	# 	model.eval()

	# 	# bleu
	# 	hyp_corpus = []
	# 	ref_corpus = []

	# 	evaliter = iter(dataset.iter_loader)
	# 	out_count = 0

	# 	with torch.no_grad():
	# 		for idx in range(len(evaliter)):
	# 			batch_items = evaliter.next()

	# 			# load data
	# 			batch_src_ids = batch_items['src_ids'][0]
	# 			batch_src_att_mask = batch_items['src_att_mask'][0]
	# 			batch_tgt_ids = batch_items['tgt_ids'][0]
	# 			batch_tgt_seqs = batch_items['tgt_seqs']

	# 			# separate into minibatch
	# 			batch_size = batch_src_ids.size(0)
	# 			n_minibatch = int(batch_size / self.minibatch_size)
	# 			n_minibatch += int(batch_size % self.minibatch_size > 0)

	# 			for bidx in range(n_minibatch):

	# 				i_start = bidx * self.minibatch_size
	# 				i_end = min(i_start + self.minibatch_size, batch_size)
	# 				src_ids = batch_src_ids[i_start:i_end]
	# 				src_att_mask = batch_src_att_mask[i_start:i_end]
	# 				tgt_ids = batch_tgt_ids[i_start:i_end]
	# 				tgt_seqs = [elem[0] for elem in batch_tgt_seqs[i_start:i_end]]

	# 				# import pdb; pdb.set_trace()
	# 				preds = model.forward_eval(src_ids, src_att_mask, max_length=100)

	# 				# print - both are lists
	# 				out_count = self._print_hyp(out_count, tgt_seqs, preds)

	# 				# accumulate corpus
	# 				for sidx in range(len(preds)):
	# 					hyp_corpus.append(preds[sidx].split())
	# 					ref_corpus.append([tgt_seqs[sidx].split()])

	# 	# import pdb; pdb.set_trace()
	# 	bleu = torchtext.data.metrics.bleu_score(hyp_corpus, ref_corpus)
	# 	metrics = {}
	# 	metrics['bleu'] = bleu

	# 	return metrics

	def count_edits(self,input, prediction,remove_punct=False):
		'''
		Count number of edits
		'''
		if prediction[-2:] == ' .':
			prediction = prediction[:-2]+'.'
		if input[-2:] == ' .':
			input = input[:-2]+'.'
		exclude = set(string.punctuation)
		if remove_punct:
			# remove punctuation
			input = ''.join(ch for ch in input if ch not in exclude)
			prediction = ''.join(ch for ch in prediction if ch not in exclude)
		annotator = errant.load('en')
		input = annotator.parse(input)
		prediction = annotator.parse(prediction)
		alignment = annotator.align(input, prediction)
		edits = annotator.merge(alignment)
		return len(edits)

	def _train_batch(self, model, batch_items,noise_configs):

		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>

			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# load data
		batch_src_ids = batch_items['src_ids'][0]
		batch_src_att_mask = batch_items['src_att_mask'][0]
		batch_tgt_ids = batch_items['tgt_ids'][0]

		# separate into minibatch
		batch_size = batch_src_ids.size(0)
		n_minibatch = int(batch_size / self.minibatch_size)
		n_minibatch += int(batch_size % self.minibatch_size > 0)

		# loss
		resloss = 0
		noise_bar = torch.tensor(np.zeros(self.embedding_dim)).to(device=self.device)

		for bidx in range(n_minibatch):
			# load data
			i_start = bidx * self.minibatch_size
			i_end = min(i_start + self.minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_att_mask = batch_src_att_mask[i_start:i_end]
			tgt_ids = batch_tgt_ids[i_start:i_end]

			# Forward propagation
			if "dversarial" in noise_configs['noise_type']:
				model.eval()
				outputs = model.forward_train(src_ids, src_att_mask, tgt_ids, noise_configs, self.noise)
				loss = outputs.loss
				loss /= n_minibatch

				grad = torch.autograd.grad(loss, self.noise, retain_graph=True, create_graph=False)[0]
				pdb.set_trace()
				if "norm" in noise_configs['noise_type']:
					print("norm")
					with torch.no_grad():
						for i in range(len(src_ids)):
							grad[i] /= (torch.norm(grad[i]) + 1e-10)
				new_noise = self.noise + self.alpha * grad
				with torch.no_grad():
					for i in range(len(src_ids)):
						new_noise[i] /= (torch.norm(new_noise[i]) + 1e-10)
				new_noise *= self.weight
				# pdb.set_trace()
				

				with torch.no_grad():
					preds, scores = model.forward_translate(src_ids=src_ids, src_att_mask=src_att_mask, max_length=self.max_tgt_len, mode=self.mode, noise_config=noise_configs, grad_noise=new_noise)

				self.final_pred.append(preds)
				# Update the noise to be the noise bar
				noise_bar += torch.sum(new_noise, dim=(0,1))
			else:
				outputs = model.forward_train(src_ids, src_att_mask, tgt_ids, noise_configs, self.noise)
				loss = outputs.loss
				loss /= n_minibatch

		with torch.no_grad():
			noise_bar /= batch_size
			self.noise = self.gamma * self.noise + (1 - self.gamma) * noise_bar.expand([self.minibatch_size,self.seq_length,self.embedding_dim])
		# print(torch.mean(self.noise))
		# print(torch.var(self.noise))
		# print(noise_bar.max())
		
		self.noise.requires_grad = True
		return resloss


	def _train_epochs(self,
		train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		step = start_step
		step_elapsed = 0
		prev_bleu = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# loop over epochs
		# for epoch in tqdm(range(start_epoch, n_epochs + 1)):
		
		outdir="/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v005/eval-clc-test-beam-1/2_"+self.noise_configs['noise_type']+"_add_"+str(self.noise_configs['weight'])
		if not os.path.isdir(outdir):
			os.makedirs(outdir)
		with open(os.path.join(outdir, 'translate.txt'), 'w', encoding="utf8") as f:
			# update lr
			if self.lr_warmup_steps != 0:
				self.optimizer.optimizer = self.lr_scheduler(
					self.optimizer.optimizer, step, init_lr=self.lr_init,
					peak_lr=self.lr_peak, warmup_steps=self.lr_warmup_steps)

			# print lr
			# for param_group in self.optimizer.optimizer.param_groups:
			# 	log.info('epoch:{} lr: {}'.format(epoch, param_group['lr']))
			# 	lr_curr = param_group['lr']

			# construct batches - allow re-shuffling of data
			log.info('--- construct train set ---')
			train_set.construct_batches(is_train=False)
			if dev_set is not None:
				log.info('--- construct dev set ---')
				dev_set.construct_batches(is_train=False)

			# print info
			steps_per_epoch = len(train_set.iter_loader)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))

			# log.info(" ---------- Epoch: %d, Step: %d ----------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			log.info('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# loop over batches
			model.train(True)
			trainiter = iter(train_set.iter_loader)
			for idx in tqdm(range(steps_per_epoch)):
				# with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
				# load batch items
				batch_items = trainiter.next()

				# update macro count
				step += 1
				step_elapsed += 1

				if self.lr_warmup_steps != 0:
					self.optimizer.optimizer = self.lr_scheduler(
						self.optimizer.optimizer, step, init_lr=self.lr_init,
						peak_lr=self.lr_peak, warmup_steps=self.lr_warmup_steps)

				# Get loss
				loss = self._train_batch(model, batch_items, self.noise_configs)
				# print_loss_total += loss
			count = 0
			for sidx in range(len(self.final_pred)):
				for item in self.final_pred[sidx]:
					f.write('{}\n'.format(item))
					count += 1
			# pdb.set_trace()
			print(count)
				# if step % self.print_every == 0 and step_elapsed > self.print_every:
				# 	print_loss_avg = print_loss_total / self.print_every
				# 	print_loss_total = 0

				# 	log_msg = 'Progress: %d%%, Train nlll: %.4f' % (
				# 		step / total_steps * 100, print_loss_avg)

				# 	log.info(log_msg)
				# 	self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)

				# # Checkpoint
				# if (step != 0 and step % self.checkpoint_every == 0) or step == total_steps:

				# 	# save criteria
				# 	if dev_set is not None:
				# 		metrics = self._evaluate_batches(model, dev_set)

				# 		bleu = metrics['bleu']
				# 		log_msg = 'Progress: %d%%, Dev bleu: %.4f' % (step / total_steps * 100, bleu)
				# 		log.info(log_msg)
				# 		self.writer.add_scalar('dev_bleu', bleu, global_step=step)

				# 		# save condition
				# 		cond_bleu = (bleu < 0.1) or (prev_bleu <= bleu)
				# 		save_cond = cond_bleu

				# 		if save_cond:
				# 			# save best model
				# 			ckpt = Checkpoint(model=model,
				# 					   optimizer=self.optimizer,
				# 					   epoch=epoch, step=step)

				# 			saved_path = ckpt.save(self.expt_dir)
				# 			log.info('saving at {} ... '.format(saved_path))
				# 			# reset
				# 			prev_bleu = bleu
				# 			count_no_improve = 0
				# 			count_num_rollback = 0
				# 		else:
				# 			count_no_improve += 1

				# 		# roll back
				# 		if count_no_improve > self.max_count_no_improve:
				# 			# no roll back - break after self.max_count_no_improve epochs
				# 			if self.max_count_num_rollback == 0:
				# 				break
				# 			# resuming
				# 			latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
				# 			if type(latest_checkpoint_path) != type(None):
				# 				resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
				# 				log.info('epoch:{} step: {} - rolling back {} ...'.format(
				# 					epoch, step, latest_checkpoint_path))
				# 				model = resume_checkpoint.model
				# 				self.optimizer = resume_checkpoint.optimizer
				# 				# A walk around to set optimizing parameters properly
				# 				resume_optim = self.optimizer.optimizer
				# 				defaults = resume_optim.param_groups[0]
				# 				defaults.pop('params', None)
				# 				defaults.pop('initial_lr', None)
				# 				self.optimizer.optimizer = resume_optim.__class__(
				# 					model.parameters(), **defaults)

				# 			# reset
				# 			count_no_improve = 0
				# 			count_num_rollback += 1

				# 		# update learning rate
				# 		if count_num_rollback > self.max_count_num_rollback:

				# 			# roll back
				# 			latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
				# 			if type(latest_checkpoint_path) != type(None):
				# 				resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
				# 				log.info('epoch:{} step: {} - rolling back {} ...'.format(
				# 					epoch, step, latest_checkpoint_path))
				# 				model = resume_checkpoint.model
				# 				self.optimizer = resume_checkpoint.optimizer
				# 				# A walk around to set optimizing parameters properly
				# 				resume_optim = self.optimizer.optimizer
				# 				defaults = resume_optim.param_groups[0]
				# 				defaults.pop('params', None)
				# 				defaults.pop('initial_lr', None)
				# 				self.optimizer.optimizer = resume_optim.__class__(
				# 					model.parameters(), **defaults)

				# 			# decrease lr
				# 			for param_group in self.optimizer.optimizer.param_groups:
				# 				param_group['lr'] *= 0.5
				# 				lr_curr = param_group['lr']
				# 				log.info('reducing lr ...')
				# 				log.info('step:{} - lr: {}'.format(step, param_group['lr']))

				# 			# check early stop
				# 			if lr_curr <= 0.125 * self.lr_peak:
				# 				log.info('early stop ...')
				# 				break

				# 			# reset
				# 			count_no_improve = 0
				# 			count_num_rollback = 0

				# 		model.train(mode=True)
				# 		if ckpt is None:
				# 			ckpt = Checkpoint(model=model,
				# 					   optimizer=self.optimizer,
				# 					   epoch=epoch, step=step)
				# 		ckpt.rm_old(self.expt_dir, keep_num=self.keep_num)
				# 		log.info('n_no_improve {}, num_rollback {}'.format(
				# 			count_no_improve, count_num_rollback))

				# 	sys.stdout.flush()
			
			# else:
			# 	if dev_set is None:
			# 		# save every epoch if no dev_set
			# 		ckpt = Checkpoint(model=model,
			# 				   optimizer=self.optimizer,
			# 				   epoch=epoch, step=step)
			# 		saved_path = ckpt.save_epoch(self.expt_dir, epoch)
			# 		log.info('saving at {} ... '.format(saved_path))
			# 		continue

			# 	else:
			# 		continue

			# # break nested for loop
			# break


	def train(self, train_set, model, num_epochs=5, optimizer=None,
		dev_set=None, grab_memory=True):

		"""
			Run training for a given model.
			Args:
				train_set: dataset
				dev_set: dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run
				resume(bool, optional): resume training with the latest checkpoint
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training

			Returns:
				model (seq2seq.models): trained model.
		"""

		if 'resume' in self.load_mode:

			assert type(self.load_dir) != type(None)

			# resume training
			latest_checkpoint_path = self.load_dir
			self.logger.info('{} {} ...'.format(self.load_mode, latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			self.logger.info(model)
			self.optimizer = resume_checkpoint.optimizer
			if self.optimizer is None:
				self.optimizer = Optimizer(torch.optim.Adam(model.parameters(),
					lr=self.lr_init), max_grad_norm=self.max_grad_norm)

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			for name, param in model.named_parameters():
				log = self.logger.info('resuming {}:{}'.format(name, param.size()))

			# start from prev
			start_epoch = resume_checkpoint.epoch # start from the saved epoch!
			step = resume_checkpoint.step# start from the saved step!

		elif 'restart' in self.load_mode:

			assert type(self.load_dir) != type(None)

			# resume training
			latest_checkpoint_path = self.load_dir
			self.logger.info('{} {} ...'.format(self.load_mode, latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model

			self.logger.info(model)
			self.optimizer = resume_checkpoint.optimizer
			if self.optimizer is None:
				self.optimizer = Optimizer(torch.optim.Adam(model.parameters(),
					lr=self.lr_init), max_grad_norm=self.max_grad_norm)

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			for name, param in model.named_parameters():
				log = self.logger.info('restarting {}:{}'.format(name, param.size()))
				param.requires_grad = True

			# just for the sake of finetuning
			start_epoch = 1
			step = 0

		else:
			start_epoch = 1
			step = 0
			self.logger.info(model)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model.parameters(),
					lr=self.lr_init), max_grad_norm=self.max_grad_norm)
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s" %(self.optimizer.optimizer))

		# reserve memory
		# import pdb; pdb.set_trace()
		if self.device == torch.device('cuda') and grab_memory:
			reserve_memory(device_id=self.gpu_id)

		self._train_epochs(train_set, model, num_epochs, start_epoch, step, dev_set=dev_set)

		return model
