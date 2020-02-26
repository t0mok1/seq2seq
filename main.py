import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import configparser
from pathlib import Path

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
from utils.logger import getMyLogger
from utils.loader import MyDataset, get_vocab

def main():
	config = configparser.ConfigParser()
	config.read('config/default.conf', encoding='utf-8')
	defaults = config['Defaults']
	batchsize = defaults.get('--batchsize')
	dataDir = defaults.get('--data_dir')
	outputDir = defaults.get('--out')

	logger = getMyLogger(outputDir)
	logger.debug('batchsize : {}'.format(batchsize))
	logger.debug('output_dir : {}'.format(outputDir))

	trainSrcPath = Path(dataDir).joinpath('train_20000.txt.en')
	trainTrgPath = Path(dataDir).joinpath('train_20000.txt.ja')
	logger.debug('trainSrcPath : {}'.format(trainSrcPath))
	logger.debug('trainTrgPath : {}'.format(trainTrgPath))

	srcVocab, trgVocab = get_vocab(trainSrcPath, trainTrgPath, src_freq=7, trg_freq=10)

	trainDataset = MyDataset(trainSrcPath, trainTrgPath, srcVocab, trgVocab)

if __name__ == '__main__':
	main()
