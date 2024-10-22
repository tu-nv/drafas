# Variables
REGISTRY = nvantu

# Pattern rule for building, tagging, and pushing Docker images
.PHONY: build
build-and-push:
	docker build -t $(IMAGE):latest docker/$(IMAGE) && \
	docker tag $(IMAGE):latest $(REGISTRY)/$(IMAGE):latest && \
	docker push $(REGISTRY)/$(IMAGE):latest

# Targets that specify which images to build
drafas-ollama:
	$(MAKE) build-and-push IMAGE=drafas-ollama

drafas-triton:
	$(MAKE) build-and-push IMAGE=drafas-triton

