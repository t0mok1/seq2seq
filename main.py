import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import configparser
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
from utils.logger import get_logger

def main():
	config = configparser.ConfigParser()
	config.read('config/default.conf', encoding='utf-8')
	defaults = config['Defaults']
	batchsize = defaults.get('--batchsize')
	data_dir = defaults.get('--data_dir')
	output_dir = defaults.get('--out')

	logger = get_logger(output_dir)
	logger.debug('batchsize : {}'.format(batchsize))
	logger.debug('data_dir : {}'.format(data_dir))
	logger.debug('output_dir : {}'.format(output_dir))

if __name__ == '__main__':
	main()
