# -*- coding: utf-8 -*-
"""A parser and 2 class to settle all configurations

This module includes 1 parser and 2 class to settle all configurations.
Some values in class will depend on parser value (i.e. dir names following
taskname)

Notes:
	config => task related settings, directories
	args => model parameters
	opt => training options

Example:
	configurations('Name of this task')
Todo:
    * create function to refresh all connected values upon change.

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import argparse
import time

def computer_time():
	time_list = time.ctime().split()
	hr_min = ''.join(time_list[3].split(':')[:2])
	return '_'.join(time_list[1:3])+'_'+hr_min

class model_param():
	def __init__(self, config):
		if config.datatype =='mnist':
			self.img_channel_num = 1
		else:
			self.img_channel_num = 3
		
		#trainer
		self.lr = 0.0002
		self.betas = (0.5, 0.999)

		self.img_size = 64
		self.batch_size = 128

# trainer setting
class training_param():
	def __init__(self, config):
		#train setting
		self.epochs = 5

		self.task_dir = config.task_result_root+'/'+config.taskname
		self.dir_list = [self.task_dir]

		self.model_filepath = self.task_dir+'/model.t7'


def configurations(taskname=None, datatype='mnist'):

	# determine datatype (will effect model)
	parser = argparse.ArgumentParser(description='test')
	parser.add_argument('--taskname', type=str, default=taskname)
	parser.add_argument('--datatype', type=str, default=datatype,
						help='choose: mnist, lsun, celeba')


	# permanent directories
	dir_args = parser.add_argument_group('directories')
	dir_args.add_argument('--data_dir_root', type=str, default='/home/jimmy/datastore',
		help='root for data download spot')
	dir_args.add_argument('--task_result_root', type=str, default='./output')
	# task related directories in training_param

	config, unparsed = parser.parse_known_args()

	if config.taskname is None and taskname is None:
		print('WARANING: USING COMPUTER TIME FOR TASKNAME')
		time.sleep(2.1)
		config.taskname = 'T'+computer_time()
	else:
		config.taskname = taskname

	print('taskname is:', config.taskname)
	print('datatype is:', config.datatype)
	args = model_param(config)
	opt = training_param(config)
	return config, args, opt

