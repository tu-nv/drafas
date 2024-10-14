# Variables
REGISTRY = nvantu


# Build the Docker image
.PHONY: build
drafas-ollama:
	docker build -t drafas-ollama:latest docker/drafas-ollama && \
	docker tag drafas-ollama:latest $(REGISTRY)/drafas-ollama:latest && \
	docker push $(REGISTRY)/drafas-ollama:latest
