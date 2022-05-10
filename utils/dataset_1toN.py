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

# customised
from utils.misc import check_device

import logging
logging.basicConfig(level=logging.INFO)

class IterDataset(torch.utils.data.Dataset):

	"""
		load features from

		'src_seqs':src_seqs[i_start:i_end],
		'tgt_seqs':tgt_seqs[i_start:i_end],
		'tgt_sids':tgt_ids[i_start:i_end],
	"""

	def __init__(self, batches, device):

		super(Dataset_1toN).__init__()

		self.task_prefix = "translate English to German: "
		self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
		self.device = device

		self.batches = batches

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		# import pdb; pdb.set_trace()

		src_seqs = self.batches[index]['src_seqs'] # lis
		tgt_seqs = self.batches[index]['tgt_seqs'] # lis
		tgt_sids = self.batches[index]['tgt_sids'] # lis

		# src id + mask
		src_encoding = self.t5_tokenizer(
			[self.task_prefix + seq for seq in src_seqs],
			padding='longest',
			return_tensors="pt")
		src_ids = src_encoding.input_ids # b x len
		src_attention_mask = src_encoding.attention_mask # b x len

		# tgt id
		tgt_encoding = self.t5_tokenizer(
			tgt_seqs,
			padding='longest',
			return_tensors="pt")
		tgt_ids = tgt_encoding.input_ids # b x len

		# replace pad with -100, not account for loss
		tgt_ids = [[(tgt_id if tgt_id != self.t5_tokenizer.pad_token_id else -100)
			for tgt_id in tgt_ids_example] for tgt_ids_example in tgt_ids]
		tgt_ids = torch.tensor(tgt_ids) # b x len

		# lengths
		src_lens = torch.sum(src_encoding.attention_mask, dim=1)
		tgt_lens = torch.sum(tgt_encoding.attention_mask, dim=1)

		# import pdb; pdb.set_trace()
		batch = {
			# for model use
			'src_ids': src_ids.to(device=self.device), # tensor
			'src_att_mask': src_attention_mask.to(device=self.device), # tensor
			'tgt_ids': tgt_ids.to(device=self.device), # tensor
			# for output use
			'tgt_seqs': tgt_seqs, # lis - for bleu calculation
			'sent_ids': tgt_sids, # lis
			'src_lens': src_lens,
			'tgt_lens': tgt_lens
		}

		return batch


class Dataset_1toN(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params
		path_src,
		path_tgt,
		batch_size=64,
		use_gpu=True,
		logger=None
		):

		super(Dataset_1toN, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.batch_size = batch_size

		self.use_gpu = use_gpu
		self.device = check_device(self.use_gpu)

		self.logger = logger
		if type(self.logger) == type(None):
			self.logger = logging.getLogger(__name__)

		self.load_sentences()
		self.pre_process()


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()
		with codecs.open(self.path_tgt, encoding='UTF-8') as f:
			self.tgt_sentences = f.readlines()

		self.num_sentences_src = len(self.src_sentences)
		self.num_sentences_tgt = len(self.tgt_sentences)
		assert self.num_sentences_tgt % self.num_sentences_src == 0
		self.gennum = self.num_sentences_tgt // self.num_sentences_src


	def pre_process(self):

		self.src_seqs = []
		self.tgt_seqs = []
		self.tgt_sids = []

		# src
		"""
		back in new york i am the head of development for a non profit called robin hood
		when i'm not fighting poverty i'm fighting fires as the assistant
		"""
		for idx in range(self.num_sentences_src):
			src_sent = self.src_sentences[idx].strip()
			self.src_seqs.extend([src_sent] * self.gennum)

		# tgt
		"""
		ted_01096_0012610_0016680-000+000       -0.180077       Zurück in New York bin ich der Leiter der Entwicklung für eine gemeinnützige Organisation namens Robin Hood.
		ted_01096_0012610_0016680-000+001       -0.183871       Zurück in New York bin ich der Entwicklungsleiter für eine gemeinnützige Organisation namens Robin Hood.
		"""
		for idx in range(self.num_sentences_tgt):
			elems = self.tgt_sentences[idx].strip().split('\t')
			tgtid = elems[0]
			if len(elems) == 2:
				tgtsemt = ''
			else:
				tgtsent = elems[2]
			self.tgt_sids.append(tgtid)
			self.tgt_seqs.append(tgtsent)

		assert len(self.src_seqs) == len(self.tgt_seqs)
		self.num_sentences = len(self.tgt_seqs)
		# import pdb; pdb.set_trace()


	def construct_batches(self, is_train=False):

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
				'tgt_seqs':self.tgt_seqs[i_start:i_end],
				'tgt_sids':self.tgt_sids[i_start:i_end],
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.device)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
