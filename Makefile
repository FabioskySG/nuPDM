USER_NAME := nupdm
TAG_NAME := v1
IMAGE_NAME := nupdm

WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
UID := $(shell id -u)
GID := $(shell id -g)

create_entrypoint:
	echo '#!/bin/bash' > entrypoint.sh
	echo 'source /home/$(USER_NAME)/workspace/.env' >> entrypoint.sh
	echo 'exec bash' >> entrypoint.sh
	chmod +x entrypoint.sh

define run_docker
	docker run -it --rm \
		--net host \
		--gpus all \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(IMAGE_NAME)_container \
		-u $(USER_NAME) \
		-v ./:/home/$(USER_NAME)/workspace \
		-v /home/robesafe/Datasets/nuscenes:/home/$(USER_NAME)/Datasets/nuscenes \
		-v /home/robesafe/Datasets/PDM_Lite_FSG:/home/$(USER_NAME)/Datasets/PDM_Lite_FSG \
		-v /media/robesafe/SSD_Samsung_1TB/nuPDM:/home/$(USER_NAME)/Datasets/nuPDM \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e DISPLAY=$(DISPLAY) \
		-e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
		--env-file .env \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(XDG_RUNTIME_DIR):$(XDG_RUNTIME_DIR) \
		'$(IMAGE_NAME)':$(TAG_NAME) \
		./entrypoint.sh
endef

build: create_entrypoint
	docker build . -t '$(IMAGE_NAME)':$(TAG_NAME) --force-rm --build-arg USER=$(USER_NAME) --build-arg USER_ID=$(UID) --build-arg USER_GID=$(GID)

run:
	$(call run_docker)

attach:
	docker exec -it $(IMAGE_NAME)_container bash

	

