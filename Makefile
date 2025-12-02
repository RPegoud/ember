BASE_FLAGS=-it --rm  --shm-size=1g -v ${PWD}:/home/app/ember -w /home/app/ember

DOCKER_IMAGE_NAME = ember
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN = docker run $(BASE_FLAGS) $(IMAGE)

build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

run:
	$(DOCKER_RUN) python $(example)

bash:
	$(DOCKER_RUN) bash