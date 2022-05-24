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
from utils.gec_tools import get_sentences

import logging
logging.basicConfig(level=logging.INFO)
import pdb

class IterDataset(torch.utils.data.Dataset):

	"""
		load features from

		'src_seqs':src_seqs[i_start:i_end],
	"""

	def __init__(self, batches, max_src_len, device):

		super(Dataset_EVAL).__init__()

		self.task_prefix = "gec: "
		# self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
		correction_model_tag = "zuu/grammar-error-correcter"
		self.t5_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
		self.max_src_len = max_src_len
		self.device = device

		self.batches = batches

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		# import pdb; pdb.set_trace()

		src_seqs = self.batches[index]['src_seqs'] # lis

		# src id + mask
		src_encoding = self.t5_tokenizer(
			[self.task_prefix + seq for seq in src_seqs],
			padding='longest',
			max_length=self.max_src_len,
			truncation=True,
			return_tensors="pt")
		src_ids = src_encoding.input_ids # b x len
		src_attention_mask = src_encoding.attention_mask # b x len
		# pdb.set_trace()
		batch = {
			'src_ids': src_ids.to(device=self.device), # tensor
			'src_att_mask': src_attention_mask.to(device=self.device), # tensor
		}

		return batch


class Dataset_EVAL(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params
		path_src,
		max_src_len=32,
		batch_size=64,
		use_gpu=True,
		logger=None
		):

		super(Dataset_EVAL, self).__init__()

		self.path_src = path_src
		self.max_src_len = max_src_len
		self.batch_size = batch_size

		self.use_gpu = use_gpu
		self.device = check_device(self.use_gpu)

		self.logger = logger
		if type(self.logger) == type(None):
			self.logger = logging.getLogger(__name__)

		self.load_sentences()
		# self.load_seqences()


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()

		self.num_sentences = len(self.src_sentences)
		self.src_seqs = [sentence.strip() for sentence in self.src_sentences]
	
	def load_seqences(self):
		# Load input sentences
		_, self.src_sentences = get_sentences(self.path_src)
		self.num_sentences = len(self.src_sentences)
		self.src_seqs = [sentence.strip() for sentence in self.src_sentences]


	def construct_batches(self):

		# record #sentences
		self.logger.info("num sentences: {}".format(self.num_sentences))

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_sentences / self.batch_size +
			(self.num_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_sentences)
			batch = {
				'src_seqs':self.src_seqs[i_start:i_end],
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': False,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.max_src_len, self.device)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
