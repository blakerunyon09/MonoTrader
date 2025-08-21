# Updates data for a given stock symbol
# Usage: make update ARGS="<SYMBOL>"
update:
	python -m src.data.update $(ARGS)

# Trains a model for a given stock symbol
# Usage: make train ARGS="<SYMBOL>"
train:
	python -m pipelines.train_bag_learner $(ARGS)