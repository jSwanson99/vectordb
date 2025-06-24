.PHONY: build start stop logs clean dev

IMAGE_NAME=vectordb-test
IMAGE_TAG=dev

start:
	docker compose up --build -d
	make logs

dev:
	docker compose up --watch

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

rebuild:
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

