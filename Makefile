APP_NAME=sawseen/pytorch_cv:pose_forecaster
CONTAINER_NAME=pose_forecaster

run: ## Run container
	docker run \
	    --runtime=nvidia \
		-itd \
		--name=${CONTAINER_NAME} \
        -v $(shell pwd):/pose_forecaster \
		$(APP_NAME) bash

exec: ## Run a bash in a running container
	docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}