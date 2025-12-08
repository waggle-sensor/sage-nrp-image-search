.PHONY: down build calculate load get

# vars
NETWORK_NAME=weaviate_network
app_image=inquire_benchmark
weavloader_image=inquire_weavloader
INQUIRE_DATASET=sagecontinuum/INQUIRE-Benchmark-small
IMAGE_BATCH_SIZE=25
QUERY_BATCH_SIZE=5
SAMPLE_SIZE=0
WORKERS=5
IMAGE_RESULTS_FILE=image_search_results.csv
QUERY_EVAL_METRICS_FILE=query_eval_metrics.csv

down:

	# Stop and remove app container
	@if docker ps -a --format '{{.Names}}' | grep -q "^$(app_image)$$"; then \
		echo "Stopping $(app_image)..."; \
		docker stop $(app_image); \
		echo "Removing $(app_image)..."; \
		docker rm $(app_image); \
	else \
		echo "$(app_image) does not exist. Skipping..."; \
	fi

	# Stop and remove app container
	@if docker ps -a --format '{{.Names}}' | grep -q "^$(weavloader_image)$$"; then \
		echo "Stopping $(weavloader_image)..."; \
		docker stop $(weavloader_image); \
		echo "Removing $(weavloader_image)..."; \
		docker rm $(weavloader_image); \
	else \
		echo "$(weavloader_image) does not exist. Skipping..."; \
	fi

# Build custom services
build:
	# Build App image
	docker build -t $(app_image) ./app

	# Build Weaviate Loader
	docker build -t $(weavloader_image) ./weavloader

#Calculate INQUIRE benchmark
calculate:

	# Stop and remove app container
	@if docker ps -a --format '{{.Names}}' | grep -q "^$(app_image)$$"; then \
		echo "Stopping $(app_image)..."; \
		docker stop $(app_image); \
		echo "Removing $(app_image)..."; \
		docker rm $(app_image); \
	else \
		echo "$(app_image) does not exist. Skipping..."; \
	fi

	# Run app container with the network configuration
	docker run --name $(app_image) --network $(NETWORK_NAME) -p 7862:7862 --restart on-failure \
		-e PYTHONUNBUFFERED=1 \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e INQUIRE_DATASET='$(INQUIRE_DATASET)' \
		-e QUERY_BATCH_SIZE='$(QUERY_BATCH_SIZE)' \
		-e IMAGE_RESULTS_FILE='$(IMAGE_RESULTS_FILE)' \
		-e QUERY_EVAL_METRICS_FILE='$(QUERY_EVAL_METRICS_FILE)' \
		-e CLUSTER_FLAG='True' \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-d $(app_image)

# load in INQUIRE dataset into weaviate
load:

	# Stop and remove app container
	@if docker ps -a --format '{{.Names}}' | grep -q "^$(weavloader_image)$$"; then \
		echo "Stopping $(weavloader_image)..."; \
		docker stop $(weavloader_image); \
		echo "Removing $(weavloader_image)..."; \
		docker rm $(weavloader_image); \
	else \
		echo "$(weavloader_image) does not exist. Skipping..."; \
	fi

	# Run data loader
	docker run --name $(weavloader_image) --network $(NETWORK_NAME) --restart on-failure \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e INQUIRE_DATASET='$(INQUIRE_DATASET)' \
		-e IMAGE_BATCH_SIZE='$(IMAGE_BATCH_SIZE)' \
		-e SAMPLE_SIZE='$(SAMPLE_SIZE)' \
		-e WORKERS='$(WORKERS)' \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-d $(weavloader_image)

# retrieve the results
get:
	# get the result files
	docker cp $(app_image):/app/$(IMAGE_RESULTS_FILE) .
	docker cp $(app_image):/app/$(QUERY_EVAL_METRICS_FILE) .
