# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Detect GPU availability on host to decide docker run args
HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_NVIDIA),1)
DOCKER_GPU := --gpus all
else
DOCKER_GPU :=
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif


build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t gsa:v0 .
run:
	# Download/check SAM checkpoint (resume if partial)
ifeq (,$(wildcard ./sam_vit_h_4b8939.pth))
	curl -fL -C - -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth || \
		( rm -f sam_vit_h_4b8939.pth && curl -fL -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth )
endif
	# Download/check GroundingDINO checkpoint (resume if partial)
ifeq (,$(wildcard ./groundingdino_swint_ogc.pth))
	curl -fL -C - -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth || \
		( rm -f groundingdino_swint_ogc.pth && curl -fL -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth )
endif
	docker run $(DOCKER_GPU) -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/Grounded-Segment-Anything \
	-e DISPLAY=$DISPLAY \
	--name=gsa \
	--ipc=host -it gsa:v0
