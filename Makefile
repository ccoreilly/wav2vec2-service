VERSION ?= 0.1.0

build:
	docker build . -t ghcr.io/ccoreilly/wav2vec2-catala:${VERSION}

push: build
	docker push ghcr.io/ccoreilly/wav2vec2-catala:${VERSION}

deploy:
	kustomize build k8s | kubectl apply -f -

undeploy:
	kustomize build k8s | kubectl delete -f -