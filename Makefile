project_name = project_name

apt-packages = git
install-requirements:
	sudo -E apt update -y
	sudo -E apt install -y $(apt-packages)
	sudo -E pip install --upgrade pip
	sudo -E pip install -r requirements.txt

train-mnist: install-requirements
	PYTHONPATH=$(shell pwd):$(PYTHONPATH) \
	python $(project_name)/train.py \
	--hash=$(shell git rev-parse HEAD)

docker-image = yoshum/pytorch:latest-py3.6-cuda10.0-ubuntu18.04

host-src := $(shell pwd)
host-data := $(shell realpath ./experiments/data)
host-results := $(shell realpath ./experiments/results)

common-options := -it --rm --runtime=nvidia \
	-v $(host-src):/workspace/src \
	-v $(host-data):/workspace/data \
	-v $(host-results):/workspace/results \
	-e LOCAL_USER=$(shell id -un) \
	-e LOCAL_UID=$(shell id -u) \
	-e LOCAL_GID=$(shell id -g)

target = train-mnist
train-target: ## execute "make $(target)" in a docker container
		docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		/bin/bash -c "cd /workspace/src; make $(target)"

run-container: ## run a docker container
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		/bin/bash


tb-port = 63000
logdir = /workspace/results/logs
tensorboard: ## launch TensorBoard in a docker container
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		--port $(tb-port):$(tb-port) \
		/bin/bash -c "cd /workspace/src; make _run-tensorboard tb-port=$(tb-port) logdir=$(logdir)"

_run-tensorboard: install-requirements
	sudo -E pip install tensorboard
	tensorboard --logdir=$(logdir) --port=$(tb-port)


jupy-port = 63001
jupyter: ## launch Jupyter Notebook in a docker container
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		--port $(jupy-port):$(jupy-port) \
		/bin/bash -c "cd /workspace/src; make _run-jupyter jupy-port=$(jupy-port)"

_run-jupyter: install-requirements
	sudo -E pip install jupyter
	cd /workspace; jupyter notebook --no-browser --ip=0.0.0.0 --port=$(jupy-port)


.PHONY: help
 
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help