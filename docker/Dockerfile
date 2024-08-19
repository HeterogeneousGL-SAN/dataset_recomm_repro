# Use an image with GPU support and CUDA Toolkit pre-installed
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip

# Set the working directory in the container
COPY . /code/
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -U "ray[data,train,tune,serve]"
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN pip3 install torch_geometric
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz
RUN pip3 install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip3 install  dglgo -f https://data.dgl.ai/wheels-test/repo.html





