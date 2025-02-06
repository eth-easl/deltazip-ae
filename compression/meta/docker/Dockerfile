FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV MAX_JOBS=16
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6"
COPY . /app
WORKDIR /app
RUN pip install -r /app/requirements.txt
RUN pip install --no-deps .
RUN pip install triton
RUN mv /usr/local/cuda/include/cufile.h /tmp/cufile.h && git clone --branch branch-23.10 https://github.com/rapidsai/kvikio.git && cd kvikio && ./build.sh kvikio && cd .. && rm -rf kvikio && mv /tmp/cufile.h /usr/local/cuda/include/cufile.h