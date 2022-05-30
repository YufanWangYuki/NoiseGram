import random
import os
import numpy as np
import sys
sys.path.append(r"/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.misc import set_global_seeds, save_config, validate_config, check_device

# huggingface api
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings
warnings.filterwarnings("ignore")

import pdb
import data_helpers

from translate_gramformer import correct

class Seq2seq(nn.Module):

	""" T5 enc-dec model """

	def __init__(self):

		super(Seq2seq, self).__init__()

		# model_name = "prithivida/grammar_error_correcter_v1"
		model_name = "zuu/grammar-error-correcter"
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name) #T5ForConditionalGeneration


	def forward_train(self, src_ids, src_att_mask, tgt_ids, noise_config, grad_noise=None):

		"""
			for training

			args:
				src: [src1, src2, ..]
					e.g. src1 = "Welcome to NYC"
				tgt: [tgt1, tgt2, ..]
					e.g. tgt1 = "Bienvenue à NYC"

			outputs: (listed in order)
				loss, logits, past_key_values, decoder_hidden_states,
				decoder_attentions, cross_attentions,
				encoder_last_hidden_states, encoder_hidden_states,
				encoder_attentions
			use as:
				loss = outputs.loss

		"""

		inputs_embeds = self.model.encoder.embed_tokens(src_ids)
		embedding_dim = inputs_embeds.shape[2]
		device = inputs_embeds.device
		noise = data_helpers.add_noise(src_ids, embedding_dim, random_type=noise_config['noise_type'], 
                    word_keep=noise_config['word_keep'], weight=noise_config['weight'], mean=noise_config['mean'],
					replace_map=noise_config['replace_map'],grad_noise=grad_noise)
		if not torch.is_tensor(noise):
			noise = noise.astype(np.float32)
			noise = torch.tensor(noise).to(device=device)
		else:
			noise = noise.float()
		if noise_config['noise_way'] == 'mul':
			new_embeds = inputs_embeds * noise[:len(inputs_embeds),:len(inputs_embeds[0]),:]
		elif noise_config['noise_way'] == 'add':
			new_embeds = inputs_embeds + noise[:len(inputs_embeds),:len(inputs_embeds[0]),:]
		
		outputs = self.model(
			# input_ids=src_ids,
			attention_mask=src_att_mask,
			labels=tgt_ids,
			inputs_embeds=new_embeds
		)

		return outputs


	def forward_eval(self, src_ids, src_att_mask, max_length=100):

		"""
			for inference
		"""

		# import pdb; pdb.set_trace()

		outputs = self.model.generate(
			input_ids=src_ids,
			attention_mask=src_att_mask,
			max_length=max_length,
			do_sample=False,
			length_penalty = 1.0,
			early_stopping = True,
			use_cache = True
		)

		outseqs = self.tokenizer.batch_decode(outputs,
			skip_special_tokens=True, clean_up_tokenization_spaces=True)

		return outseqs


	def forward_translate(self, src_ids, src_att_mask, max_length=100, mode='beam-1', noise_config=None, grad_noise=None):

		"""
			for inference
			mode:
				beam-1: beam=1, greedy
				beam-50: beam=50
				sample: topK sampling
		"""

		import pdb; pdb.set_trace()

		gen_mode = mode.split('-')[0] # beam-N, sample-N, beamdiv-N
		num = int(mode.split('-')[-1])
		
		if noise_config is not None:
			inputs_embeds = self.model.encoder.embed_tokens(src_ids)
			embedding_dim = inputs_embeds.shape[2]
			device = inputs_embeds.device
			grad_noise=grad_noise
			# pdb.set_trace()
			# if noise_config['noise'] == 2 and 'Adversarial' in noise_config['noise_type']:
			# 	grad_noise=Nonec
			new_embeds = inputs_embeds

			if noise_config['noise'] == 2:
				noise = data_helpers.add_noise(src_ids, embedding_dim, random_type=noise_config['noise_type'], 
                    word_keep=noise_config['word_keep'], weight=noise_config['weight'], mean=noise_config['mean'],
					replace_map=noise_config['replace_map'],grad_noise=grad_noise)
				if 'cuda' not in noise.device:
					noise = torch.tensor(noise.astype(np.float32)).to(device=device)
				if noise_config['noise_way'] == 'mul':
					new_embeds = inputs_embeds * noise
				elif noise_config['noise_way'] == 'add':
					new_embeds = inputs_embeds + noise

		
			# pdb.set_trace()
			encoder_outputs = self.model.encoder(attention_mask=src_att_mask,inputs_embeds=new_embeds)
			if gen_mode == 'beam':
				outputs = self.model.generate(
					# input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					num_beams=num,
					num_return_sequences=num,
					do_sample=False,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True, # output scores as well as predictions
					output_scores=True,
					encoder_outputs=encoder_outputs,
					bos_token_id=0
				)

			elif gen_mode == 'beamdiv':
				# beam search with diversity penalty
				outputs = self.model.generate(
					input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					num_beams=num,
					num_return_sequences=num,
					do_sample=False,
					num_beam_groups=5,
					diversity_penalty=0.5,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True, # output scores as well as predictions
					output_scores=True
				)

			elif gen_mode == 'sample':
				# not very diverse if doing sampling
				outputs = self.model.generate(
					input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					do_sample=True,
					num_return_sequences=num,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True
					# output_scores=True
				)

			outseqs = self.tokenizer.batch_decode(outputs.sequences,
				skip_special_tokens=True, clean_up_tokenization_spaces=True)

			# import pdb; pdb.set_trace()
			if gen_mode == 'beam' or gen_mode == 'beamdiv':
				if num == 1:
					# obtain seq score from per word scores
					prep_scores = torch.stack(outputs.scores, dim=1).softmax(-1) # B*N x len x #vocab
					prep_seqs = outputs.sequences[:,1:] # B*N x len
					# word logp -> seq logp
					select = torch.gather(prep_scores, 2, prep_seqs.unsqueeze(-1)).squeeze(-1)
					# length norm
					mask = (prep_seqs != 0)
					scores = 1. * torch.sum(torch.log(select) * mask, dim=1) / torch.sum(mask, dim=1)
				else:
					scores = outputs.sequences_scores

			elif gen_mode == 'sample':
				scores = [0] * len(outseqs)

		else:
			if gen_mode == 'beam':
				outputs = self.model.generate(
					input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					num_beams=num,
					num_return_sequences=num,
					do_sample=False,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True, # output scores as well as predictions
					output_scores=True
				)

			elif gen_mode == 'beamdiv':
				# beam search with diversity penalty
				outputs = self.model.generate(
					input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					num_beams=num,
					num_return_sequences=num,
					do_sample=False,
					num_beam_groups=5,
					diversity_penalty=0.5,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True, # output scores as well as predictions
					output_scores=True
				)

			elif gen_mode == 'sample':
				# not very diverse if doing sampling
				outputs = self.model.generate(
					input_ids=src_ids,
					attention_mask=src_att_mask,
					max_length=max_length,
					do_sample=True,
					num_return_sequences=num,
					length_penalty=1.0,
					early_stopping=True,
					use_cache=True,
					return_dict_in_generate=True
					# output_scores=True
				)

			outseqs = self.tokenizer.batch_decode(outputs.sequences,
				skip_special_tokens=True, clean_up_tokenization_spaces=True)

			# import pdb; pdb.set_trace()
			if gen_mode == 'beam' or gen_mode == 'beamdiv':
				if num == 1:
					# obtain seq score from per word scores
					prep_scores = torch.stack(outputs.scores, dim=1).softmax(-1) # B*N x len x #vocab
					prep_seqs = outputs.sequences[:,1:] # B*N x len
					# word logp -> seq logp
					select = torch.gather(prep_scores, 2, prep_seqs.unsqueeze(-1)).squeeze(-1)
					# length norm
					mask = (prep_seqs != 0)
					scores = 1. * torch.sum(torch.log(select) * mask, dim=1) / torch.sum(mask, dim=1)
				else:
					scores = outputs.sequences_scores

			elif gen_mode == 'sample':
				scores = [0] * len(outseqs)
		# pdb.set_trace()
		


		# if sent != outseqs[0]:
		# 	pdb.set_trace()
		# if 'invented' in outseqs[0]:
		# 	pdb.set_trace()
		# 	correction_model_tag = "zuu/grammar-error-correcter"
		# 	tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
		# 	test_model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)
		# 	test_model.to(device)
		return outseqs, scores


	def forward_genatt(self, src_ids, src_att_mask, tgt_ids):

		"""
			for attention generation
		"""

		# import pdb; pdb.set_trace()

		outputs = self.model(
			input_ids=src_ids,
			attention_mask=src_att_mask,
			labels=tgt_ids,
			output_attentions=True,
			return_dict=True
		)

		return outputs


	def forward_translate_greedy(self, src_ids, src_att_mask, max_length=100):

		"""
			greedy decoding - with per word probability
		"""

		# import pdb; pdb.set_trace()

		outputs = self.model.generate(
			input_ids=src_ids,
			attention_mask=src_att_mask,
			max_length=max_length,
			num_beams=1,
			num_return_sequences=1,
			do_sample=False,
			length_penalty=1.0,
			early_stopping=True,
			use_cache=True,
			return_dict_in_generate=True, # output scores as well as predictions
			output_scores=True
		)

		outseqs = self.tokenizer.batch_decode(outputs.sequences,
			skip_special_tokens=True, clean_up_tokenization_spaces=True)

		# import pdb; pdb.set_trace()
		# obtain per word scores
		prep_scores = torch.stack(outputs.scores, dim=1).softmax(-1) # B*N x len x #vocab
		prep_seqs = outputs.sequences[:,1:] # B*N x len
		# word probs
		select = torch.gather(prep_scores, 2, prep_seqs.unsqueeze(-1)).squeeze(-1)
		pdf = torch.distributions.categorical.Categorical(logits=torch.log(prep_scores))
		entropy = pdf.entropy()	# b x len

		# length mask
		mask = (prep_seqs != 0)
		probs = select * mask
		entropy = entropy * mask
		ids = outputs.sequences[:,1:]
		tokens = [self.tokenizer.convert_ids_to_tokens(elem) for elem in ids]

		return outseqs, tokens, probs, entropy
	
	def correct(self, input_sentence, max_candidates=1):
		correction_prefix = "gec: "
		input_sentence = correction_prefix + input_sentence
		input_ids = self.tokenizer.encode(input_sentence, return_tensors='pt')
		
		preds = self.model.generate(
            input_ids,
            do_sample=True, 
            max_length=128, 
            top_k=50, 
            top_p=0.95, 
            early_stopping=True,
            num_return_sequences=max_candidates)
		
		corrected = set()
		for pred in preds:  
			corrected.add(self.tokenizer.decode(pred, skip_special_tokens=True).strip())
		
		corrected = list(corrected)
		return corrected
