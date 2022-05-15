import torch
torch.cuda.empty_cache()
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset_1toN import Dataset_1toN
from utils.dataset_eval import Dataset_EVAL
from utils.dataset import Dataset
from utils.misc import save_config, validate_config
from utils.misc import get_memory_alloc, log_ckpts, plot_attention_transformer
from utils.misc import plot_alignment, check_device, combine_weights
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

logging.basicConfig(level=logging.INFO)
import pdb

def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, required=False, default=None, help='test src dir')
	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--combine_path', type=str, default='None', help='combine multiple ckpts if given dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--mode', type=str, default='beam-1', help='beam-N | sample')
	parser.add_argument('--max_tgt_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--fold', type=int, default=0, help='nth split of corpora')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')

	# noise
	parser.add_argument('--noise', type=int, default=1,
		help='1 for without noise 2 for using noise')
	parser.add_argument('--ntype', type=str, default='Gaussian',help='noise type')
	parser.add_argument('--nway', type=str, default='mul',help='noise add way: mul or add')
	parser.add_argument('--mean', type=float, default=1.0,help='noise mean')
	parser.add_argument('--weight', type=float, default=0.0,help='noise weight')
	parser.add_argument('--word_keep', type=float, default=1.0,help='word keep')
	parser.add_argument('--replace_map', type=list, default=None,
		help='replace map')
	return parser

def translate(test_set, model, test_path_out, max_tgt_len, mode, device, noise_config):

	"""
		run inference, output be:
		<txt0-0>
		<txt0-1>
		...
		<txt0-N>
		<txt1-0>
		...
	"""
	# import pdb; pdb.set_trace()
	# pdb.set_trace()
	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):
				if idx > 1000:
					break
				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['src_ids'][0]
				src_att_mask = batch_items['src_att_mask'][0]

				time1 = time.time()
				preds, scores = model.forward_translate(src_ids, src_att_mask,
					max_length=max_tgt_len, mode=mode, noise_config=noise_config)
				time2 = time.time()
				print('comp time: ', time2-time1)

				# import pdb; pdb.set_trace()
				for sidx in range(len(preds)):
					f.write('{}\n'.format(preds[sidx]))


def translate_verbo(test_set, model, test_path_out, max_tgt_len, mode, device):

	"""
		run inference, output be:
		<id> <score> <txt0-0>
		<id> <score> <txt0-1>
		...
		<id> <score> <txt0-N>
		<id> <score> <txt1-0>
		...
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	sentcount = 0
	with open(os.path.join(test_path_out, 'translate.txt.verbo'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['src_ids'][0]
				src_att_mask = batch_items['src_att_mask'][0]

				time1 = time.time()
				preds, scores = model.forward_translate(src_ids, src_att_mask,
					max_length=max_tgt_len, mode=mode)
				time2 = time.time()
				print('comp time: ', time2-time1)

				# import pdb; pdb.set_trace()
				genN = int(mode.split('-')[-1])
				for sidx in range(len(preds)):
					sentid = sidx // genN + sentcount
					sampid = sidx % genN
					score = scores[sidx]
					pred = preds[sidx]
					f.write('sent_{:08d}-{:03d}\t{:.6f}\t{}\n'.format(
						sentid, sampid, score, pred))
				sentcount += len(preds) // genN


def translate_verbo_fold(test_set, model, test_path_out, max_tgt_len, mode, fold, device):

	"""
		run inference, output be:
		<id> <score> <txt0-0>
		<id> <score> <txt0-1>
		...
		<id> <score> <txt0-N>
		<id> <score> <txt1-0>
		...
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	# =============================================
	# for fold
	NSAMPLE = 50
	BATCHSIZE = 1
	FOLDNUM = 500000
	idx_st = FOLDNUM * fold
	idx_ed = min(FOLDNUM * (fold + 1), len(evaliter))
	print('num batches: {}/{}, st-ed:{}-{}'.format(idx_ed - idx_st, len(evaliter), idx_st, idx_ed))

	# =============================================
	# pick up from where dropped off - temporary fix
	fdir = os.path.join(test_path_out, 'translate-{:03d}.txt.verbo'.format(fold))
	if os.path.exists(fdir):
		f = open(os.path.join(test_path_out, 'translate-{:03d}.txt.verbo'.format(fold)), 'r', encoding="utf8")
		lines = f.readlines()
		# access last line
		line = lines[-1]
		sid = line.strip().split()[0] # 'sent_000003005-049'
		hypnum = int(sid.split('_')[1].split('-')[1]) # 049
		scount = int(sid.split('_')[1].split('-')[0]) # 000003005
		sinit = int(lines[0].strip().split()[0].split('_')[1].split('-')[0])
		import pdb; pdb.set_trace()
		# check no residual from next incomplete batch
		assert hypnum == NSAMPLE - 1	# since hypnum starts from 0
		assert (scount + 1) % BATCHSIZE == 0 	# since scount starts from 0
		# reset idx_st
		bcount = FOLDNUM * fold + (scount - sinit + 1) // BATCHSIZE
		idx_st = bcount
		# reset to append to file
		sentcount = scount + 1
		f = open(os.path.join(test_path_out, 'translate-{:03d}.txt.verbo'.format(fold)), 'a', encoding="utf8")
	else:
		sentcount = idx_st
		f = open(os.path.join(test_path_out, 'translate-{:03d}.txt.verbo'.format(fold)), 'w', encoding="utf8")

	# =============================================
	# generation
	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			batch_items = evaliter.next()
			print('{}/{}, st-ed:{}-{}'.format(idx, len(evaliter), idx_st, idx_ed))
			if idx < idx_st or idx >= idx_ed:
				print('pass')
				continue

			# load data
			src_ids = batch_items['src_ids'][0]
			src_att_mask = batch_items['src_att_mask'][0]

			time1 = time.time()
			preds, scores = model.forward_translate(src_ids, src_att_mask,
				max_length=max_tgt_len, mode=mode)
			time2 = time.time()
			print('comp time: ', time2-time1)

			# import pdb; pdb.set_trace()
			genN = int(mode.split('-')[-1])
			for sidx in range(len(preds)):
				sentid = sidx // genN + sentcount
				sampid = sidx % genN
				score = scores[sidx]
				pred = preds[sidx]
				f.write('sent_{:08d}-{:03d}\t{:.6f}\t{}\n'.format(
					sentid, sampid, score, pred))
			sentcount += len(preds) // genN


def gen_att(test_set, model, test_path_out, mode, device):

	"""
		attention map generation (given both En-De)
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	# mkdir
	fdir = os.path.join(test_path_out, 'attmap')
	if not os.path.exists(fdir): os.makedirs(fdir)
	plotdir = os.path.join(test_path_out, 'attplot')
	if not os.path.exists(plotdir): os.makedirs(plotdir)

	# ---------------- temperary fix -----------------
	# assume batch_size == 50
	# check exist
	flogpath = os.path.join(fdir, 'attention.log')
	if os.path.exists(flogpath):
		flog = open(flogpath, 'r')
		lines = flog.readlines()
		# access last line
		line = lines[-1].strip()
		# ted_02246_0649580_0657080-samp000+024   ted-190053.npy  49      41      33
		elems = line.split()
		sid = elems[0] # 'ted_02246_0649580_0657080-samp000+024'
		fid = elems[1] # 'ted-190053.npy'
		nidx = elems[2]
		assert nidx == '49'
		scount = int(fid.split('.')[0].split('-')[1]) + 1 # 190053+1
		# import pdb; pdb.set_trace()
		# if exist, then append
		flog = open(flogpath, 'a')
	else:
		scount = 0
		flog = open(flogpath, 'w')
		flog.write('{}\t\t\t\t\t{}\t\t{}\t{}\t{}\n'.format(
			'sent_id', 'loc', 'idx', 'len_out', 'len_in'))

	# --------------- save to npy ------------------
	plotcount = 0
	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			if idx < scount:
				print('pass', idx, len(evaliter))
				continue

			print(idx, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['src_ids'][0]
			src_att_mask = batch_items['src_att_mask'][0]
			tgt_ids = batch_items['tgt_ids'][0]
			sent_ids = batch_items['sent_ids']
			src_lens = batch_items['src_lens'][0]
			tgt_lens = batch_items['tgt_lens'][0]

			outputs = model.forward_genatt(src_ids, src_att_mask, tgt_ids)

			# batch x Layer_dec x Nhead x Len_out x Len_in
			attmap_raw = torch.stack(outputs.cross_attentions, dim=0).transpose(0,1)
			# mean over head -> batch x Layer_dec x Len_out x Len_in
			attmap_ave = torch.mean(attmap_raw, dim=2)
			# mean over dec -> batch x Len_out x Len_in
			attmap_aveave = torch.mean(attmap_ave, dim=1)

			# only save part of the attmap! (save space...)
			fname = 'ted-{:06}.npy'.format(idx)
			fpath = os.path.join(fdir, fname)
			# import pdb; pdb.set_trace()

			if mode == 'ave_head':
				np.save(fpath, attmap_ave.cpu().detach().numpy())
			elif mode == 'ave_head_dec':
				np.save(fpath, attmap_aveave.cpu().detach().numpy())

			for sidx in range(len(sent_ids)):
				# write to flog
				sent_id = sent_ids[sidx][0]
				len_out = int(tgt_lens[sidx])
				len_in = int(src_lens[sidx])
				flog.write('{}\t{}\t{}\t{}\t{}\n'.format(sent_id, fname, sidx, len_out, len_in))

			# plot samples
			if plotcount < 5 and scount == 0:

				sent_id = sent_ids[0][0]
				len_in = int(src_lens[0])
				len_out = int(tgt_lens[0])
				attelem = attmap_ave.cpu().detach().numpy()[0][:,:len_out,:len_in]
				plotpath = os.path.join(plotdir, sent_id)
				plot_attention_transformer(attelem, plotpath)
				plotcount += 1

			sys.stdout.flush()


		flog.close()


def translate_perword(test_set, model, test_path_out, max_tgt_len, device):

	"""
		run inference, output be:
		<s>
		<word> <score>
		<word> <score>
		...
		<s>
		...
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	BOS_TOK = '▁' # used for tokenisation
	PAD_TOK = '<pad>'
	EOS_TOK = '</s>'
	UNK_TOK = '<unk>'

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['src_ids'][0]
				src_att_mask = batch_items['src_att_mask'][0]

				time1 = time.time()
				preds, tokens, probs, entropy = model.forward_translate_greedy(
					src_ids, src_att_mask, max_length=max_tgt_len)

				# # == debug ==
				# preds_v2, scores_v2 = model.forward_translate(src_ids, src_att_mask,
				# 	max_length=max_tgt_len, mode='beam-1')
				# import pdb; pdb.set_trace()
				# # ===========

				time2 = time.time()
				print('comp time: ', time2-time1)

				# import pdb; pdb.set_trace()
				for sidx in range(len(preds)):
					# loop over sentences
					# combine token-level probs, entropy to word level
					wordlis = []
					problis = []
					hlis = []

					wordelem = ''
					probelem = []
					helem = []
					for tidx in range(len(tokens[0])):
						# loop over tokens
						token = tokens[sidx][tidx]
						prob = probs[sidx][tidx]
						h = entropy[sidx][tidx]

						if token == EOS_TOK:
							break
						elif token == UNK_TOK:
							continue
						elif token == PAD_TOK:
							continue
						elif token[0] == BOS_TOK:
							# clear prev word
							if wordelem != '':
								wordlis.append(wordelem)
								problis.append(1. * sum(probelem) / len(probelem))
								hlis.append(1. * sum(helem) / len(helem))
							# new word
							wordelem = token
							probelem = [prob]
							helem = [h]
						else:
							# append to prev
							wordelem += token
							probelem.append(prob)
							helem.append(h)

					# last word
					if wordelem != '':
						wordlis.append(wordelem)
						problis.append(1. * sum(probelem) / len(probelem))
						hlis.append(1. * sum(helem) / len(helem))

					# import pdb; pdb.set_trace()
					# print to file
					f.write('<s>\n')
					for widx in range(len(wordlis)):
						word = wordlis[widx][1:]
						prob = problis[widx]
						h = hlis[widx]
						f.write('{}\t{:.5f}\t{:.5f}\n'.format(word, prob, h))
					f.write('<s>\n\n')


def translate_perword_teacherforcing(test_set, model, test_path_out, max_tgt_len, device):

	"""
		run inference, output be:
		<s>
		<word> <score>
		<word> <score>
		...
		<s>
		...
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	# set vocab
	from transformers import AutoTokenizer
	model_name = "prithivida/grammar_error_correcter_v1"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	BOS_TOK = '▁' # used for tokenisation
	PAD_TOK = '<pad>'
	EOS_TOK = '</s>'
	UNK_TOK = '<unk>'

	ftxt = open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8")
	ftsv = open(os.path.join(test_path_out, 'translate.tsv'), 'w', encoding="utf8")

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['src_ids'][0]
			src_att_mask = batch_items['src_att_mask'][0]
			tgt_ids = batch_items['tgt_ids'][0]
			tgt_seqs = batch_items['tgt_seqs']

			# import pdb; pdb.set_trace()
			# run teacher forcing
			res = model.forward_train(src_ids, src_att_mask, tgt_ids)
			res_logits = res.logits
			# entropy
			res_pdf = torch.distributions.categorical.Categorical(logits=res_logits)
			entropy = res_pdf.entropy()	# b x len
			# choose probability
			res_probs = torch.softmax(res_logits, dim=-1)
			tgt_ids[tgt_ids==-100] = 3
			probs = torch.gather(res_probs, 2, tgt_ids.unsqueeze(-1)).squeeze(-1)
			# get tokens
			tokens = [tokenizer.convert_ids_to_tokens(elem) for elem in tgt_ids]

			for sidx in range(len(tokens)):
				# loop over sentences
				# combine token-level probs, entropy to word level
				wordlis = []
				problis = []
				hlis = []

				wordelem = ''
				probelem = []
				helem = []
				for tidx in range(len(tokens[0])):
					# loop over tokens
					token = tokens[sidx][tidx]
					prob = probs[sidx][tidx]
					h = entropy[sidx][tidx]

					if token == EOS_TOK:
						break
					elif token == UNK_TOK:
						continue
					elif token == PAD_TOK:
						continue
					elif token[0] == BOS_TOK:
						# clear prev word
						if wordelem != '':
							wordlis.append(wordelem)
							problis.append(1. * sum(probelem) / len(probelem))
							hlis.append(1. * sum(helem) / len(helem))
						# new word
						wordelem = token
						probelem = [prob]
						helem = [h]
					else:
						# append to prev
						wordelem += token
						probelem.append(prob)
						helem.append(h)

				# last word
				if wordelem != '':
					wordlis.append(wordelem)
					problis.append(1. * sum(probelem) / len(probelem))
					hlis.append(1. * sum(helem) / len(helem))

				# import pdb; pdb.set_trace()
				# print to file
				ftsv.write('<s>\n')
				for widx in range(len(wordlis)):
					word = wordlis[widx][1:]
					prob = problis[widx]
					h = hlis[widx]
					ftsv.write('{}\t{:.5f}\t{:.5f}\n'.format(word, prob, h))
				ftsv.write('<s>\n\n')

				sent = [word[1:] for word in wordlis]
				ftxt.write('{}\n'.format(' '.join(sent)))

	ftxt.close()
	ftsv.close()


def translate_save_logp(test_set, model, test_path_out, max_tgt_len, device):

	"""
		teacher forcing mode
		save topK logp
	"""
	# import pdb; pdb.set_trace()

	# load test
	test_set.construct_batches()
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	f = open(os.path.join(test_path_out, 'translate.log'), 'w')

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['src_ids'][0]
			src_att_mask = batch_items['src_att_mask'][0]
			tgt_ids = batch_items['tgt_ids'][0]
			tgt_seqs = batch_items['tgt_seqs']

			res = model.forward_train(
				src_ids, src_att_mask, tgt_ids)

			import pdb; pdb.set_trace()
			logits = res.logits
			probs = torch.softmax(logits, dim=-1)
			topk = torch.topk(probs, k=50)
			# not yet completed


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	test_path_out = config['test_path_out']
	load_dir = config['load']
	max_tgt_len = config['max_tgt_len']
	batch_size = config['batch_size']
	mode = config['mode']
	use_gpu = config['use_gpu']
	fold = config['fold']

	noise_configs = {
			'noise':config['noise'],
			'noise_type':config['ntype'],
			'weight':config['weight'],
			'mean':config['mean'],
			'word_keep':config['word_keep'],
			'replace_map':config['replace_map'],
			'noise_way':config['nway']
		}
	# if config['noise'] == 1:
	# 	noise_configs=None
	
	# set test mode
	# 1: save comb ckpt
	# 2: save to state dict - for loading model else where

	MODE = config['eval_mode']
	if MODE != 1 and MODE != 2:
		if not os.path.exists(test_path_out):
			os.makedirs(test_path_out)
		config_save_dir = os.path.join(test_path_out, 'eval.cfg')
		save_config(config, config_save_dir)

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# load test_set
	if MODE == 6 or MODE == 8:
		# 1 src -> N tgt
		test_set = Dataset_1toN(test_path_src, test_path_tgt,
					batch_size=batch_size, use_gpu=use_gpu)
	else:
		test_set = Dataset_EVAL(test_path_src,
					max_src_len=900, batch_size=batch_size, use_gpu=use_gpu)

	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()
	# run eval
	if MODE == 3:
		# run inference
		translate(test_set, model, test_path_out, max_tgt_len, mode, device, noise_configs)

	elif MODE == 4:
		# run inference - with id + score
		translate_verbo(test_set, model, test_path_out, max_tgt_len, mode, device)

	elif MODE == 5:
		# run inference - with id + score - fold
		translate_verbo_fold(test_set, model, test_path_out, max_tgt_len, mode, fold, device)

	elif MODE == 6:
		# generate attention map
		gen_att(test_set, model, test_path_out, mode, device)

	elif MODE == 7:
		# run inference - with word id + per word score
		translate_perword(test_set, model, test_path_out, max_tgt_len, device)

	elif MODE == 8:
		# run inference - with word id + per word score
		translate_perword_teacherforcing(test_set, model, test_path_out, max_tgt_len, device)

	elif MODE == 1:
		# save combined model
		assert type(config['combine_path']) != type(None)
		model = combine_weights(config['combine_path'])

		ckpt = Checkpoint(model=model,
				   optimizer=None, epoch=0, step=0)
		saved_path = ckpt.save_customise(
			os.path.join(config['combine_path'].strip('/')+'-combine','combine'))
		log_ckpts(config['combine_path'], config['combine_path'].strip('/')+'-combine')
		print('saving at {} ... '.format(saved_path))

	elif MODE == 2:
		# save model to state_dict, allow access by keys without matching dir
		modeldir = os.path.join(load_dir, 'model-dict.pt')
		torch.save(model.state_dict(), modeldir)
		print('saved to {} ...'.format(modeldir))


if __name__ == '__main__':
	main()
