FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL author="Juanwu Lu" email="juanwu@purdue.edu" env="stable"
RUN mkdir -p /app/seneva
WORKDIR /app/seneva
COPY . .
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    git \
    wget \
    && apt-get clean
# Install Rust environment
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.75.0 -y
ENV PATH="/root/.cargo/bin:${PATH}"
# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -i https://download.pytorch.org/whl/cu118 torch==2.3.0
RUN pip3 install git+https://github.com/argoverse/av2-api#egg=av2
RUN pip3 install .
