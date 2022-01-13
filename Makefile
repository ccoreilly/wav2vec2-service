VERSION ?= 0.1.1

build:
	docker build . -t ghcr.io/ccoreilly/wav2vec2-catala:${VERSION}

push: build
	docker push ghcr.io/ccoreilly/wav2vec2-catala:${VERSION}

build-onnx:
	docker build . -f onnx.Dockerfile -t ghcr.io/ccoreilly/wav2vec2-catala-onnx:${VERSION}

push-onnx: build-onnx
	docker push ghcr.io/ccoreilly/wav2vec2-catala-onnx:${VERSION}

deploy:
	kustomize build k8s | kubectl apply -f -

undeploy:
	kustomize build k8s | kubectl delete -f -