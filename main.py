import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import configparser
from pathlib import Path
import torch.utils.data as data

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
from utils.logger import getMyLogger
from utils.loader import MyDataset, get_vocab, collate_fn

def main():
	config = configparser.ConfigParser()
	config.read('config/default.conf', encoding='utf-8')
	defaults = config['Defaults']
	batchsize = defaults.getint('--batchsize')
	dataDir = defaults.get('--data_dir')
	outputDir = defaults.get('--out')

	logger = getMyLogger(outputDir)
	logger.debug('batchsize : {}'.format(batchsize))
	logger.debug('output_dir : {}'.format(outputDir))

	trainSrcPath = Path(dataDir).joinpath('train_20000.txt.en')
	trainTrgPath = Path(dataDir).joinpath('train_20000.txt.ja')
	logger.debug('trainSrcPath : {}'.format(trainSrcPath))
	logger.debug('trainTrgPath : {}'.format(trainTrgPath))

	srcVocab, trgVocab = get_vocab(trainSrcPath, trainTrgPath, src_freq=1, trg_freq=1)

	trainDataset = MyDataset(trainSrcPath, trainTrgPath, srcVocab, trgVocab)
	trainLoader = data.DataLoader(dataset=trainDataset, batch_size=batchsize, shuffle=True, num_workers=0, collate_fn=collate_fn)

	trainMaxLength = trainLoader.dataset.max_length
	print(maxLength)

if __name__ == '__main__':
	main()
