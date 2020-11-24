cwt_generator:
	@pyenv exec python -im src.cwt_generator

test_dataset:
	@pyenv exec python -m tests.test_dataset

test_conv_ae:
	@pyenv exec python -m tests.test_conv_ae