### Project infomation

## Project name
project-name = project_name

## Default target that will be invoked by `make run` without the `target` argument
default-target = mnist

## Packages that need to be installed in a container via `apt`
# Required `pip` packages should be given in `requirements.txt`
apt-packages = 

## Name of a Docker image to use
docker-image = yoshum/pytorch:1.3-py3.6-cuda10.1-ubuntu18.04

## Add options to pass to `docker container run`
## They will be appended after the default option list defined below
docker-run-options = 

## Port number to which TensorBoard will listen
tb-port = 60000
## Port number to which Jupyter will listen
jupy-port = 61000


### targets that run an executable script in a container

mnist: _install-requirements ## run the MNIST training script (should be invoked in a container)
	PYTHONPATH=$(shell pwd):$(PYTHONPATH) \
		python scripts/train.py \
		--hash=$(shell git rev-parse HEAD)


### 

_install-requirements:
ifneq ($(apt-packages),)
	sudo -E apt update -y
	sudo -E apt install -y $(apt-packages)
endif
	python -m pip install --user --upgrade pip
	python -m pip install --user -r requirements.txt


host-src := $(shell pwd)
host-data := $(shell realpath ../data)
host-results := $(shell realpath ../results)

common-options := \
	-it --rm --runtime=nvidia \
	-v $(host-src):/workspace/src \
	-v $(host-data):/workspace/data \
	-v $(host-results):/workspace/results \
	-v /home/$(USER)/docker/home:/home/$(USER) \
	-e LOCAL_USER=$(shell id -un) \
	-e LOCAL_UID=$(shell id -u) \
	-e LOCAL_GID=$(shell id -g) \
	$(docker-run-options)


target = $(default-target)
run: ## execute "make $(target)" in a docker container
	docker container run -it --rm --runtime=nvidia \
		/bin/bash -c "cd /workspace/src; make $(target) hash='$(shell git rev-parse HEAD)' options='$(options)'"
		$(docker-image) \
		/bin/bash -c "cd /workspace/src; make $(target)"

run-remote: ## execute "make $(target)" in a docker container on $(remote-host)
	make sync-codes remote-host=$(remote-host)
	ssh -t $(remote-host) -c "cd workspace/$(project-name)/src ; make run train-target=$(target)"


sync-codes: ## sync files in the `src` directory to $(remote-host)
	rsync -e ssh -Cauvz $(host-src) $(remote-host):/home/$(USER)/workspace/$(project-name)

fetch-results: ## fetch results from $(remote-host)
	rsync -e ssh -auvz $(remote-host):/home/$(USER)/workspace/$(project-name)/results/ $(host-results)


### Targets to run a container for some specific purposes

run-container: ## run a container and execute `bash`
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		/bin/bash


logdir = /workspace/results/logs
tensorboard: ## launch TensorBoard in a docker container on port 60000
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		--port $(tb-port):$(tb-port) \
		/bin/bash -c "cd /workspace/src; make _run-tensorboard tb-port=$(tb-port) logdir=$(logdir)"

_run-tensorboard: _install-requirements
	sudo -E pip install tensorboard
	tensorboard --logdir=$(logdir) --port=$(tb-port)


jupyter: ## launch Jupyter Notebook in a docker container on port 61000
	docker container run -it --rm --runtime=nvidia \
		$(common-options) \
		$(docker-image) \
		--port $(jupy-port):$(jupy-port) \
		/bin/bash -c "cd /workspace/src; make _run-jupyter jupy-port=$(jupy-port)"

_run-jupyter: _install-requirements
	sudo -E pip install jupyter
	cd /workspace; jupyter notebook --no-browser --ip=0.0.0.0 --port=$(jupy-port)


.PHONY: help
 
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help