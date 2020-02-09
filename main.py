import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import configparser

def main():
	config = configparser.ConfigParser()
	config.read('config/default.conf', encoding='utf-8')
	defaults = config['Defaults']
	batchsize = defaults.get('--batchsize')
	data_dir = defaults.get('--data_dir')
	output_dir = defaults.get('--out')

if __name__ == '__main__':
	main()
