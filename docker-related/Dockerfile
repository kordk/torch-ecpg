# Base Image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install builder system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    software-properties-common \
    dirmngr \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Add CRAN apt repository for R 4.x
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    && apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install PyTorch from the cu124 wheel index.
# torch 2.6.0 is the LAST release published to download.pytorch.org/whl/cu124;
# 2.7.0+ ships only cu118/cu126/cu128. Do not float this pin -- an unpinned
# install silently resolves to a different CUDA build or downgrades outright.
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Copy and install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install R dependencies
COPY tools/install_dependencies.R tools/
RUN Rscript tools/install_dependencies.R

# Install source package
COPY . .
RUN pip3 install --no-cache-dir .

# Verify torch survived the subsequent pip resolves with the intended CUDA build.
# This must run AFTER `pip3 install .` -- that is the last step that can clobber torch.
RUN python3 -c "import torch; \
assert torch.__version__.startswith('2.6.0'), 'unexpected torch: ' + torch.__version__; \
assert torch.version.cuda == '12.4', 'unexpected CUDA build: ' + str(torch.version.cuda); \
print('OK torch', torch.__version__, 'cuda', torch.version.cuda)"

# Verify R installations in builder
RUN Rscript -e 'library(EpiDISH); library(sva); library(pheatmap); library(ExperimentHub); library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)'


# Runtime Stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies and R
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3 python3-pip wget curl coreutils gzip libxml2 libcurl4 libssl3 \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/cran.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy R packages resiliently (by copying parent dirs, ensuring existence)
# Copying /usr/local/lib/R and /usr/lib/R covers all standard Debian/Ubuntu R installation paths
COPY --from=builder /usr/local/lib/R /usr/local/lib/R
COPY --from=builder /usr/lib/R /usr/lib/R

# Verify R installations in runtime
RUN Rscript -e 'library(EpiDISH); library(sva); library(pheatmap); library(ExperimentHub); library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)'


# Copy application files
COPY . .

# Verify fa2 and torch survived the cross-stage copy
RUN python3 -c "import fa2" \
    && python3 -c "import torch; \
assert torch.version.cuda == '12.4', 'unexpected CUDA build: ' + str(torch.version.cuda); \
print('OK torch', torch.__version__, 'cuda', torch.version.cuda)"

# Create /work directory for volume mount
RUN mkdir -p /work
WORKDIR /work

# Rely on the pip-installed tecpg package in dist-packages; dropping PYTHONPATH=/app to avoid shadowing

# Entrypoint setup
CMD ["python3", "-m", "tecpg"]
