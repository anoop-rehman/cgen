ARG CUDA_VERSION=12.4.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git wget

RUN pip3 install -U pip setuptools wheel
RUN pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers accelerate
RUN pip3 install "PyYAML>=6.0"
RUN pip3 install --upgrade pip

# Pre-install remaining cgen dependencies to avoid conflicts during editable install
RUN pip3 install transformers>=4.24 loguru pydantic wheel

# Install vllm and its dependencies separately
RUN pip3 install vllm==0.6.3 --force-reinstall --no-deps
RUN pip3 install ray psutil sentencepiece numpy requests tqdm py-cpuinfo tokenizers protobuf aiohttp openai uvicorn pillow prometheus-client tiktoken lm-format-enforcer msgspec gguf importlib-metadata pyyaml einops nvidia-ml-py cloudpickle mistral-common datasets pyzmq

WORKDIR /workspace
COPY . /workspace/cgen

ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip3 install -e cgen/ --no-deps

RUN pip3 install flashinfer==0.1.2 -i https://flashinfer.ai/whl/cu121/torch2.4/

WORKDIR /datasets/
RUN sh /workspace/cgen/benchmark/download_sharegpt.sh /datasets/

WORKDIR /workspace/cgen/
