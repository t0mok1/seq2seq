from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

def get_logger(path):
	logger = getLogger('__name__')
	logger.setLevel(DEBUG)

	formatter = Formatter(
					'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
					datefmt='%Y/%m/%d %H:%M:%S'
				)
	stream_handler = StreamHandler()
	stream_handler.setLevel(DEBUG)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	file_handler = FileHandler(path + 'log', 'a')
	file_handler.setLevel(DEBUG)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger
