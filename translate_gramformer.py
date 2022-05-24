import sys
import os
import argparse
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


import pdb

def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_out', type=str, default='None', help='test output dir')

	# config
	parser.add_argument('--beam_width', type=int, default=1, help='beam search width')
	parser.add_argument('--device', type=str, default='cuda', help='cpu | cuda')

	return parser


def get_sentences(data_path):
	with open(data_path, 'r') as f:
		lines = f.readlines()
	texts = [' '.join(l.strip('\n').split()) for l in lines]
	return texts

def correct(model, tokenizer, sentence, beam_width, device):

	
	correction_prefix = "gec: "
	sentence = correction_prefix + sentence
	input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device=device)
	prediction_ids = model.generate(
		input_ids,
		max_length=128,
		num_beams=beam_width,
		early_stopping=True,
		num_return_sequences=1,
		do_sample=False,
		length_penalty=1.0,
		use_cache=True)

	sent = tokenizer.decode(
		prediction_ids.squeeze(),
		skip_special_tokens=True,
		clean_up_tokenization_spaces=True)
	# import pdb; pdb.set_trace()

	# input_ids = tokenizer.encode(sentence[5:], return_tensors='pt').to(device=device)
	# prediction_ids = model.generate(input_ids,max_length=128,num_beams=beam_width,early_stopping=True,num_return_sequences=1,do_sample=False,length_penalty=1.0,use_cache=True)

	# sent = tokenizer.decode(prediction_ids.squeeze(),skip_special_tokens=True,clean_up_tokenization_spaces=True)

	return sent

def correct_ids(model, tokenizer, input_ids, beam_width, device):

	# import pdb; pdb.set_trace()
	# correction_prefix = "gec: "
	# sentence = correction_prefix + sentence
	# input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device=device)
	input_ids = tokenizer.encode(sent, return_tensors='pt').to(device=config['device'])
	input_ids = input_ids.to(device=device)
	prediction_ids = model.generate(
		input_ids,
		max_length=128,
		num_beams=beam_width,
		early_stopping=True,
		num_return_sequences=1,
		do_sample=False,
		length_penalty=1.0,
		use_cache=True)

	sent = tokenizer.decode(
		prediction_ids.squeeze(),
		skip_special_tokens=True,
		clean_up_tokenization_spaces=True)

	return sent


def validate_config(config):

	for key, val in config.items():
		if isinstance(val,str):
			if val.lower() == 'true':
				config[key] = True
			if val.lower() == 'false':
				config[key] = False
			if val.lower() == 'none':
				config[key] = None

	return config


if __name__ == "__main__":

	# load config
	parser = argparse.ArgumentParser(description='Gramformer Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# Load Model and Tokenizer
	# correction_model_tag = "prithivida/grammar_error_correcter_v1"
	correction_model_tag = "zuu/grammar-error-correcter"
	tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
	model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)
	model.to(device=config['device'])

	# Load input sentences
	srcs = get_sentences(config['test_path_src'])

	# Correction (prediction) for each input sentence
	corrections = []
	for i, sent in enumerate(srcs):
		# pdb.set_trace()
		print('{}/{}'.format(i, len(srcs)))
		corrections.append(correct(model, tokenizer, sent,config['beam_width'], config['device']))
	assert len(corrections) == len(srcs), "Number of srcs don't match number of hypotheses"

	# Save predictions
	with open(config['test_path_out'], 'w') as f:
		for sent in corrections:
			f.write('{}\n'.format(sent))
