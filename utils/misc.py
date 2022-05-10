import random
import numpy as np
import psutil
import os
import torch
import torch.nn as nn

# for plot
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modules.checkpoint import Checkpoint


def reserve_memory(device_id=0):

	# import pdb; pdb.set_trace()
	total, used = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used \
		--format=csv,nounits,noheader').read().split('\n')[device_id].split(",")

	total = int(total)
	used = int(used)

	max_mem = int(total * 0.85)
	block_mem = max_mem - used

	x = torch.rand((256,1024,block_mem)).cuda()
	x = torch.rand((2,2)).cuda()


def combine_weights(path):

	"""
	 	reference - qd212
		average ckpt weights under the given path
	"""

	ckpt_path_list = [os.path.join(path, ep) for ep in os.listdir(path)]
	ckpt_state_dict_list = [Checkpoint.load(ckpt_path).model.state_dict()
		for ckpt_path in ckpt_path_list]

	model = Checkpoint.load(ckpt_path_list[0]).model
	mean_state_dict = model.state_dict()
	for key in mean_state_dict.keys():
		mean_state_dict[key] = 1. * (sum(d[key] for d in ckpt_state_dict_list)
			/ len(ckpt_state_dict_list))

	model.load_state_dict(mean_state_dict)

	return model


def log_ckpts(ckpt_path, out_path):

	f = open(os.path.join(out_path,'ckpts.log'), 'w')
	for ckpt in sorted(os.listdir(ckpt_path)):
		f.write('{}\n'.format(ckpt))
	f.close()


def check_device(use_gpu):

	""" set device """
	# import pdb; pdb.set_trace()
	# assert torch.cuda.is_available()
	if use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	return device


def get_memory_alloc():

	""" get memory used by current process """

	process = psutil.Process(os.getpid())
	mem_byte = process.memory_info().rss  # in bytes
	mem_kb = float(mem_byte) / (1024.0)
	mem_mb = float(mem_kb) / (1024.0)
	mem_gb = float(mem_mb) / (1024.0)

	return mem_kb, mem_mb, mem_gb


def get_device_memory():

	""" get total memory on current device """

	device = torch.cuda.current_device()
	mem_byte = torch.cuda.get_device_properties(device).total_memory
	mem_kb = float(mem_byte) / (1024.0)
	mem_mb = float(mem_kb) / (1024.0)
	mem_gb = float(mem_mb) / (1024.0)

	return mem_kb, mem_mb, mem_gb


def set_global_seeds(i):

	try:
		import torch
	except ImportError:
		pass
	else:
		torch.manual_seed(i)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(i)
	np.random.seed(i)
	random.seed(i)


def write_config(path, config):

	with open(path, 'w') as file:
		for x in config:
			file.write('{}={}\n'.format(x, config[x]))


def read_config(path):

	config = {}
	with open(path, 'r') as file:
		for line in file:
			x = line.strip().split('=')
			key = x[0]
			if x[1].isdigit():
				val = int(x[1])
			elif isfloat(x[1]):
				val = float(x[1])
			elif x[1].lower() == "true" or x[1].lower() == "false":
				if x[1].lower() == "true":
					val = True
				else:
					val = False
			else: # string
				val = x[1]

			config[key] = val

	return config


def print_config(config):

	print('\n-------- Config --------')
	for key, val in config.items():
		print("{}:{}".format(key, val))


def save_config(config, save_dir):

	f = open(save_dir, 'w')
	for key, val in config.items():
		f.write("{}:{}\n".format(key, val))
	f.close()


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


def _del_var(model):

	""" delete var to free up memory """

	for name, param in model.named_parameters():
		del param
	torch.cuda.empty_cache()


def plot_attention_transformer(attn, path):

	""" att: dim = Layer_dec x Layer_enc x seq_out x seq_in """

	# import pdb; pdb.set_trace()

	num_dec, len_out, len_in = attn.shape
	fig, axs = plt.subplots(num_dec // 4, 4, figsize=(12, 8))
	for dec_idx in range(num_dec):
		r = dec_idx // 4
		c = dec_idx % 4
		axs[r, c].imshow(
			attn[dec_idx], interpolation='nearest', aspect='auto')
		axs[r, c].set_title('Dec{}'.format(dec_idx))

	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)


def plot_attention_single(attn, path):

	""" att: dim = seq_out x seq_in """

	fig = plt.figure(figsize=(12, 6))
	plt.imshow(attn.T, interpolation='nearest', aspect='auto')
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)


def plot_alignment(alignment, path, src, hyp, ref=None):

	"""
		plot att alignment -
		adapted from: https://gitlab.com/Josh-ES/tacotron/blob/master/tacotron/utils/plot.py
	"""

	fig, ax = plt.subplots(figsize=(12, 10))
	im = ax.imshow(
		alignment,
		aspect='auto',
		cmap='hot',
		origin='lower',
		interpolation='none',
		vmin=0, vmax=1)
	fig.colorbar(im, ax=ax)

	plt.xticks(np.arange(len(src)), src, rotation=40)
	plt.yticks(np.arange(len(hyp)), hyp, rotation=20)

	xlabel = 'Src'
	if ref is not None:
		xlabel += '\n\nRef: ' + ' '.join(ref)

	plt.xlabel(xlabel)
	plt.ylabel('Hyp')
	plt.tight_layout()

	# save the alignment to disk
	plt.savefig(path, format='png')
