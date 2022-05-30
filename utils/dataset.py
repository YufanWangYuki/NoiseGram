# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import torch.utils.data
import collections
import codecs
import numpy as np
import random
import os

# huggingface api
from transformers import T5Tokenizer
from transformers import AutoTokenizer

# customised
from utils.misc import check_device
from utils.align_preds import align_data_train, get_sentences_dict

import logging
logging.basicConfig(level=logging.INFO)

import pdb

class IterDataset(torch.utils.data.Dataset):

	"""
		load features from

		'src_seqs':src_seqs[i_start:i_end],
		'tgt_seqs':tgt_seqs[i_start:i_end],
	"""

	def __init__(self, batches, max_src_len, max_tgt_len, device):

		super(Dataset).__init__()

		self.task_prefix = "gec: "
		# model_name = "prithivida/grammar_error_correcter_v1"
		model_name = "zuu/grammar-error-correcter"
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.max_src_len = max_src_len
		self.max_tgt_len = max_tgt_len
		self.device = device

		self.batches = batches

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		# import pdb; pdb.set_trace()

		src_seqs = self.batches[index]['src_seqs'] # lis
		tgt_seqs = self.batches[index]['tgt_seqs'] # lis

		# src id + mask
		src_encoding = self.tokenizer(
			[self.task_prefix + seq for seq in src_seqs],
			padding='longest',
			max_length=self.max_src_len,
			truncation=True,
			return_tensors="pt")
		src_ids = src_encoding.input_ids # b x len
		src_attention_mask = src_encoding.attention_mask # b x len

		# tgt id
		tgt_encoding = self.tokenizer(
			[seq for seq in tgt_seqs], # tuple to list
			padding='longest',
			max_length=self.max_tgt_len,
			truncation=True,
			return_tensors="pt")
		tgt_ids = tgt_encoding.input_ids # b x len

		# replace pad with -100, not account for loss
		tgt_ids = [[(tgt_id if tgt_id != self.tokenizer.pad_token_id else -100)
			for tgt_id in tgt_ids_example] for tgt_ids_example in tgt_ids]
		tgt_ids = torch.tensor(tgt_ids) # b x len

		batch = {
			'src_ids': src_ids.to(device=self.device), # tensor
			'src_att_mask': src_attention_mask.to(device=self.device), # tensor
			'tgt_ids': tgt_ids.to(device=self.device), # tensor
			'tgt_seqs': tgt_seqs # lis - for bleu calculation
		}

		return batch


class Dataset(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params
		path_src,
		path_tgt,
		max_src_len=32,
		max_tgt_len=32,
		batch_size=64,
		use_gpu=True,
		logger=None
		):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.max_src_len = max_src_len
		self.max_tgt_len = max_tgt_len
		self.batch_size = batch_size

		self.use_gpu = use_gpu
		self.device = check_device(self.use_gpu)

		self.logger = logger
		if type(self.logger) == type(None):
			self.logger = logging.getLogger(__name__)

		self.load_sentences()


	def load_sentences(self):
		if 'fce' in self.path_src:
			# FCE
			inc_id2text = get_sentences_dict(self.path_src)
			corr_id2text = get_sentences_dict(self.path_tgt)
			self.src_sentences, self.tgt_sentences = align_data_train(inc_id2text, corr_id2text)
			# pdb.set_trace()
		else:
			with codecs.open(self.path_src, encoding='UTF-8') as f:
				self.src_sentences = f.readlines()
			with codecs.open(self.path_tgt, encoding='UTF-8') as f:
				self.tgt_sentences = f.readlines()
		
		assert len(self.src_sentences) == len(self.tgt_sentences), \
			'Mismatch src:tgt - {}:{}'.format(len(self.src_sentences),len(self.tgt_sentences))

		self.num_sentences = len(self.src_sentences)
		self.src_seqs = [sentence.strip() for sentence in self.src_sentences]
		self.tgt_seqs = [sentence.strip() for sentence in self.tgt_sentences]
		# pdb.set_trace()


	def construct_batches(self, is_train=False):

		# record #sentences
		self.logger.info("num sentences: {}".format(self.num_sentences))

		# organise by length
		_x = list(zip(self.src_seqs, self.tgt_seqs))
		if is_train:
			# _x = sorted(_x, key=lambda l:l[1])
			random.shuffle(_x)
		src_seqs, tgt_seqs = zip(*_x)

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_sentences / self.batch_size +
			(self.num_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_sentences)
			batch = {
				'src_seqs':src_seqs[i_start:i_end],
				'tgt_seqs':tgt_seqs[i_start:i_end],
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches,
			self.max_src_len, self.max_tgt_len, self.device)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
