SHELL := /bin/bash
# .RECIPEPREFIX += . # use if you want spaces instead of tabs for Makefiles

# Type "make help" for usage instructions

TEAM ?= TRI-ML
DATA_ROOT ?= /data
TEST_PATH ?= tests/
# See Dockerfile for explanation of this variable.
SHELL_SETUP_FILE ?= /usr/local/bin/efm_env_setup.sh
# This flag is used to determine whether to run the docker commands interactively.
# This is used to allow for running in settings where we can't run interactively
# (mainly when running github actions workflows on ec2).
INTERACTIVE := yes
INITSUBMODULES := yes

reponame := $(shell basename "$(CURDIR)")
# reponame := calvin-sim-old
docker_image_name := $(reponame)
WANDB_DOCKER = $(docker_image_name)

fsx_host_dir := /<path_to>/mnt/fsx
fsx_local_dir := /opt/ml/input/data/training

DOCKER_OPTS := --rm
DOCKER_OPTS += -e XAUTHORITY -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix
DOCKER_OPTS += --shm-size 32G
DOCKER_OPTS += --ipc=host --network=host --pid=host --privileged
DOCKER_OPTS += -e AWS_DEFAULT_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET_NAME
DOCKER_OPTS += -e WANDB_API_KEY -e WANDB_DOCKER
DOCKER_OPTS += -e OPENAI_API_KEY
DOCKER_OPTS += -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility # Needed for compiling with CUDA
DOCKER_OPTS += -v ${DATA_ROOT}:/data
DOCKER_OPTS += -v $(PWD):/opt/ml/code/ -w /opt/ml/code/
DOCKER_OPTS += -v $(fsx_host_dir):$(fsx_local_dir)

ifeq ($(INTERACTIVE),yes)
  DOCKER_OPTS += -it
endif

.PHONY: help
help:
	@echo "Usage: make TARGET optional_target_arg=something"
	@echo "Available targets:"
	@echo "  configure_git gh_user=user		Configure git remotes and update the local copy"
	@echo "  install_docker		  Install Docker and NVIDIA Container Toolkit"
	@echo "  setup_docker_permissions			Setup Docker permissions"
	@echo "  xhost_local			 Allow local Docker containers to access the X server"
	@echo "  docker_build	  Build custom Docker image"
	@echo "  docker_tests			 Run tests in the docker image"
	@echo "  docker_interactive	  Run an interactive docker container"
	@echo "  python_format	  Runs black formatter on the code to check for incorrect formatting"
	@echo "  docker_run_script script=path  Run a script in a docker container"

gh_user =
.PHONY: configure_git
configure_git:
	@if [ -z "$(gh_user)" ]; then \
		echo "gh_user is not set: please specify gh_user=YOUR_GITHUB_USERNAME"; \
		exit 1; \
	fi
        # assumes you have done `git clone git@github.com:$(gh_user)/$(reponame).git` to get this Makefile
        # setup upstream
	git remote add upstream git@github.com:$(TEAM)/$(reponame).git
	git remote set-url --push upstream no_push
        # update your local copy to the latest upstream (team) version
	git fetch upstream
	git rebase upstream/main
	git submodule update --init
#	git config core.hooksPath .githooks



.SILENT: install_docker
install_docker:
        # Install docker
	if ! command -v docker &> /dev/null; then \
		curl https://get.docker.com | sh \
		  && sudo systemctl --now enable docker; \
	else \
		echo "Docker is already installed."; \
	fi
        # Install nvidia-contain-toolkit
	if ! dpkg -l | grep nvidia-container-toolkit &> /dev/null; then \
		distribution=$$(. /etc/os-release;echo $$ID$$VERSION_ID) \
		  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
		  && curl -s -L https://nvidia.github.io/libnvidia-container/$$distribution/libnvidia-container.list | \
				sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
				sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
		  && sudo apt-get update \
		  && sudo apt-get install -y nvidia-container-toolkit \
		  && sudo nvidia-ctk runtime configure --runtime=docker \
		  && sudo systemctl restart docker; \
	else \
		echo "NVIDIA container toolkit is already installed."; \
	fi

.PHONY: setup_docker_permissions
setup_docker_permissions:
        # get necessary permissions for docker
	sudo usermod -aG docker $$USER
	newgrp docker &

.PHONY: xhost_local
xhost_local:
        # Allow local Docker containers to access the X server
	xhost +local:docker

.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build --build-arg home=$(HOME) --progress=plain -f docker/Dockerfile -t $(docker_image_name):latest .
# add --no-cache to rebuild from scratch
# add --progress=plain to see the full build log

# You can't run these tests in interactive mode; it breaks the github actions workflow.
# TODO(blake.wulfe): Figure out why this "echo" is necessary for the pytest to properly run inside "tests/" only.
# This removes ".pytest_cache/" after running the tests. If you don't remove it while inside the docker
# container, it can break the github actions workflow.
.PHONY: docker_tests
docker_tests:
	docker run $(DOCKER_OPTS) --gpus all $(docker_image_name):latest bash -c 'echo && source $(SHELL_SETUP_FILE) && pytest $(TEST_PATH) && rm -rf .pytest_cache/'

.PHONY: docker_interactive
docker_interactive:
        # to get into an interactive container
	docker run $(DOCKER_OPTS) --gpus all --name $(reponame) $(docker_image_name):latest bash

.PHONY: python_format
python_format:
	docker run $(DOCKER_OPTS) --gpus all --name $(reponame) $(docker_image_name):latest black --check --verbose .

script =
.PHONY: docker_run_script
docker_run_script:
	@if [ -z "$(script)" ]; then \
		echo "Error: script is not set. Please specify script=path"; \
		exit 1; \
	fi
	docker run $(DOCKER_OPTS) --gpus all --name $(reponame) $(docker_image_name) $(script)

.PHONY: clean
clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf
