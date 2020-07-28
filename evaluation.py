#!/usr/bin/env python
import sys
import os.path
from os import walk
import scipy.stats
import numpy

tasks = ["machine-translation", "named-entity-recognition"]

#tasks = ["machine-translation", "named-entity-recognition", "question-answering", "relation-classification", "text-classification"]

def write_output_iu_triples(key, output_file, iu_f1, iu_p, iu_r, out_f1, out_p, out_r):
	if key not in iu_f1.keys():
		output_file.write(out_f1+":{0}\n".format(0.0))
		output_file.write(out_p+":{0}\n".format(0.0))
		output_file.write(out_r+":{0}\n".format(0.0))	
	else:
		output_file.write(out_f1+":{0}\n".format(iu_f1[key]))
		output_file.write(out_p+":{0}\n".format(iu_p[key]))
		output_file.write(out_r+":{0}\n".format(iu_r[key]))

def compute_recall_precision_fscore_dict(tp,fp,total):
	r = {}
	p = {}
	f1 = {}
	for key in total.keys():	
		if key not in tp.keys() and key in fp.keys():
			recall, precision, fscore = compute_recall_precision_fscore(0.0,fp[key],total[key])
		elif key in tp.keys() and key not in fp.keys():
			recall, precision, fscore = compute_recall_precision_fscore(tp[key],0.0,total[key])
		elif key not in tp.keys() and key not in fp.keys():
			recall, precision, fscore = compute_recall_precision_fscore(0.0,0.0,total[key])
		else:
			recall, precision, fscore = compute_recall_precision_fscore(tp[key],fp[key],total[key])
						
		r[key] = recall
		p[key] = precision
		f1[key] = fscore
	return r, p, f1
		
def compute_recall_precision_fscore(tp,fp,total):
	recall = tp/total
	if tp == 0.0 and fp == 0.0:
		precision = 0.0
	else:
		precision = tp/(tp+fp)
	if recall == 0.0 or precision == 0.0:
		fscore = 0.0
	else:
		fscore = (2.0*recall*precision)/(recall+precision)
	return recall, precision, fscore

def compute_total(gold):
	total = 0
	with open(gold, "rb") as f:
		content = f.readlines()
	f.close()
	gold_lines = [x.strip() for x in content]
	total = total + len(gold_lines)	
	return total

def evaluate(gold,pred):	
	total = 0
	with open(gold, "rb") as f:
		content = f.readlines()
	f.close()
	gold_lines = [x.strip() for x in content]
	total = total + len(gold_lines)	

	with open(pred, "rb") as f:
		content = f.readlines()
	f.close()
	pred_lines = [x.strip() for x in content]
	    	
	tp = 0
	tp_data = [i for i in pred_lines if i in gold_lines]
	tp = tp + len(tp_data)
	
	fp = 0
	fp_data = [i for i in pred_lines if i not in gold_lines]
	fp = fp + len(fp_data)

	fn = 0
	fn_data = [i for i in gold_lines if i not in pred_lines]
	fn = fn + len(fn_data)
	
	return total, tp, fp, fn
	   
def main(argv):
    #https://github.com/felipebravom/EmoInt/blob/master/codalab/scoring_program/
    #as per the metadata file, input and output directories are the arguments
	
	[input_dir, output_dir] = argv
	
    # unzipped submission data is always in the 'res' subdirectory
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
	gold_dir = os.path.join(input_dir, 'ref')
	submission_dir = os.path.join(input_dir, 'res')	

	# 1st error check : ensure that submissions are made for all tasks
	for task in tasks:
		task_submission_path = os.path.join(input_dir, 'res', task)
		if not os.path.exists(task_submission_path):
			sys.exit('Could not find submission file {0}'.format(task_submission_path))
			
	# evaluate for sentences, phrases, and triples
	sentences_total = 0
	sentences_tp = 0
	sentences_fp = 0
	sentences_fn = 0
	
	phrases_total = 0
	phrases_tp = 0
	phrases_fp = 0
	phrases_fn = 0	
	
	only_iu_total = 0
	only_iu_tp = 0
	only_iu_fp = 0
	only_iu_fn = 0
	
	triples_total = 0
	triples_tp = 0
	triples_fp = 0
	triples_fn = 0		
	
	iu_triples_total = {}
	iu_triples_tp = {}
	iu_triples_fp = {}
	iu_triples_fn = {}
	
	
	for task in tasks:
		
		for i in range(2):	
			#evaluate sentences	
			gold_reference_path = os.path.join(input_dir, 'ref', task, str(i), "sentences.txt")
			task_submission_path = os.path.join(input_dir, 'res', task, str(i), "sentences.txt")
			if not os.path.exists(task_submission_path):
				sys.exit('Could not find submission file {0}'.format(task_submission_path))
			total, tp, fp, fn = evaluate(gold_reference_path, task_submission_path)
			sentences_total = sentences_total + total
			sentences_tp = sentences_tp + tp
			sentences_fp = sentences_fp + fp
			sentences_fn = sentences_fn + fn
						
			#evaluate phrases
			gold_reference_path = os.path.join(input_dir, 'ref', task, str(i),  "entities.txt")
			task_submission_path = os.path.join(input_dir, 'res', task, str(i), "entities.txt")
			if not os.path.exists(task_submission_path):
				sys.exit('Could not find submission file {0}'.format(task_submission_path))
			total, tp, fp, fn = evaluate(gold_reference_path, task_submission_path)
			phrases_total = phrases_total + total
			phrases_tp = phrases_tp + tp
			phrases_fp = phrases_fp + fp
			phrases_fn = phrases_fn + fn
			
			#evaluate triples
			gold_reference_path = os.path.join(input_dir, 'ref', task, str(i), "triples")
			task_submission_path = os.path.join(input_dir, 'res', task, str(i), "triples")
			if not os.path.exists(task_submission_path):
				sys.exit('Could not find submission file {0}'.format(task_submission_path))
			for (dirpath, dirnames, filenames) in walk(gold_reference_path):
				total = 0
				tp = 0
				fp = 0
				fn = 0
				for f in filenames :
					gold_submission_iu = os.path.join(gold_reference_path, f)
					task_submission_iu = os.path.join(task_submission_path, f)
					
					key = f.replace(".txt", "")
					
					if os.path.exists(task_submission_iu):
						only_iu_total = only_iu_total + 1
						only_iu_tp = only_iu_tp + 1
					
						total_temp, tp_temp, fp_temp, fn_temp = evaluate(gold_submission_iu, task_submission_iu)
						total = total + total_temp
						tp = tp + tp_temp
						fp = fp + fp_temp
						fn = fn + fn_temp
						if key not in iu_triples_total.keys():
							iu_triples_total[key] = total_temp
						else:
							iu_triples_total[key] = iu_triples_total[key] + total_temp

						if key not in iu_triples_tp.keys():						
							iu_triples_tp[key] = tp_temp
						else:
							iu_triples_tp[key] = iu_triples_tp[key] + tp_temp

						if key not in iu_triples_fp.keys():
							iu_triples_fp[key] = fp_temp
						else:
							iu_triples_fp[key] = iu_triples_fp[key] + fp_temp							

						if key not in iu_triples_fn.keys():							
							iu_triples_fn[key] = fn_temp
						else:						
							iu_triples_fn[key] = iu_triples_fn[key] + fn_temp														
					else:
						only_iu_total = only_iu_total + 1
						only_iu_fn = only_iu_fn + 1					
					
						total_temp = compute_total(gold_submission_iu)
						fn_temp = total_temp
						
						total = total + total_temp						
						fn = fn + fn_temp
						
						if key not in iu_triples_total.keys():
							iu_triples_total[key] = total_temp
						else :
							iu_triples_total[key] = iu_triples_total[key] + total_temp

						if key not in iu_triples_fn.keys():							
							iu_triples_fn[key] = fn_temp
						else:						
							iu_triples_fn[key] = iu_triples_fn[key] + fn_temp																					
						
			for (dirpath, dirnames, filenames) in walk(task_submission_path):
				for f in filenames:
					gold_submission_iu = os.path.join(gold_reference_path, f)
					if os.path.exists(gold_submission_iu):
						continue
						
					key = f.replace(".txt", "")						

					only_iu_fp = only_iu_fp + 1
					
					task_submission_iu = os.path.join(task_submission_path, f)
					fp_temp = compute_total(task_submission_iu)
					fp = fp + fp_temp

					
					if key not in iu_triples_fp.keys():
						iu_triples_fp[key] = fp_temp
					else :
						iu_triples_fp[key] = iu_triples_fp[key] + fp_temp

			triples_total = triples_total + total
			triples_tp = triples_tp + tp
			triples_fp = triples_fp + fp
			triples_fn = triples_fn + fn
									
	sentences_r, sentences_p, sentences_f1 = compute_recall_precision_fscore(sentences_tp, sentences_fp, sentences_total)
	phrases_r, phrases_p, phrases_f1 = compute_recall_precision_fscore(phrases_tp, phrases_fp, phrases_total)
	only_iu_r, only_iu_p, only_iu_f1 = compute_recall_precision_fscore(only_iu_tp, only_iu_fp, only_iu_total)
	triples_r, triples_p, triples_f1 = compute_recall_precision_fscore(triples_tp, triples_fp, triples_total)
	iu_r, iu_p, iu_f1 = compute_recall_precision_fscore_dict(iu_triples_tp,iu_triples_fp,iu_triples_total)
	
	# the scores for the leaderboard must be in a file named "scores.txt"
	# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions	
	
	average_f1 = (sentences_f1 + phrases_f1 + only_iu_f1 + triples_f1)/4
	
	output_file=open(os.path.join(output_dir, 'scores.txt'),"w")
	output_file.write("AVG_score:{0}\n".format(average_f1))
	output_file.write("SENT_f1:{0}\n".format(sentences_f1)) 
	output_file.write("SENT_precision:{0}\n".format(sentences_p))
	output_file.write("SENT_recall:{0}\n".format(sentences_r))	
	output_file.write("ENT_SPAN_f1:{0}\n".format(phrases_f1))
	output_file.write("ENT_SPAN_precision:{0}\n".format(phrases_p))
	output_file.write("ENT_SPAN_recall:{0}\n".format(phrases_r))
	output_file.write("IU_f1:{0}\n".format(only_iu_f1))
	output_file.write("IU_precision:{0}\n".format(only_iu_p))
	output_file.write("IU_recall:{0}\n".format(only_iu_r))	
	output_file.write("TRIPLES_f1:{0}\n".format(triples_f1))
	output_file.write("TRIPLES_precision:{0}\n".format(triples_p))
	output_file.write("TRIPLES_recall:{0}\n".format(triples_r))
	write_output_iu_triples('research-problem', output_file, iu_f1, iu_p, iu_r, "RESEARCH_PROBLEM_f1", "RESEARCH_PROBLEM_precision", "RESEARCH_PROBLEM_recall")
	write_output_iu_triples('approach', output_file, iu_f1, iu_p, iu_r, "APPROACH_f1", "APPROACH_precision", "APPROACH_recall")
	write_output_iu_triples('model', output_file, iu_f1, iu_p, iu_r, "MODEL_f1", "MODEL_precision", "MODEL_recall")
	write_output_iu_triples('code', output_file, iu_f1, iu_p, iu_r, "CODE_f1", "CODE_precision", "CODE_recall")
	write_output_iu_triples('dataset', output_file, iu_f1, iu_p, iu_r, "DATASET_f1", "DATASET_precision", "DATASET_recall")
	write_output_iu_triples('experimental-setup', output_file, iu_f1, iu_p, iu_r, "EXP_SETUP_f1", "EXP_SETUP_precision", "EXP_SETUP_recall")	
	write_output_iu_triples('hyperparameters', output_file, iu_f1, iu_p, iu_r, "HYP_PARAM_f1", "HYP_PARAM_precision", "HYP_PARAM_recall")
	write_output_iu_triples('baselines', output_file, iu_f1, iu_p, iu_r, "BASELINES_f1", "BASELINES_precision", "BASELINES_recall")
	write_output_iu_triples('results', output_file, iu_f1, iu_p, iu_r, "RES_f1", "RES_precision", "RES_recall")
	write_output_iu_triples('tasks', output_file, iu_f1, iu_p, iu_r, "TASK_f1", "TASK_precision", "TASK_recall")
	write_output_iu_triples('experiments', output_file, iu_f1, iu_p, iu_r, "EXP_f1", "EXP_precision", "EXP_recall")
	write_output_iu_triples('ablation-analysis', output_file, iu_f1, iu_p, iu_r, "ABLATION_f1", "ABLATION_precision", "ABLATION_recall")
	
	
if __name__ == "__main__":
    main(sys.argv[1:])