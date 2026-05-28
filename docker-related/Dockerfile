# Base Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

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
    r-base=4.4.1-1.2204.0 \
    r-base-dev=4.4.1-1.2204.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install PyTorch with cu118
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install R dependencies
COPY tools/install_dependencies.R tools/
RUN Rscript tools/install_dependencies.R

# Install source package
COPY . .
RUN pip3 install --no-cache-dir .

# Verify R installations in builder
RUN Rscript -e 'library(EpiDISH); library(sva); library(pheatmap); library(ExperimentHub); library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)'


# Runtime Stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies and R
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    curl \
    coreutils \
    gzip \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/cran.list \

    && apt-get update && apt-get install -y --no-install-recommends \
    r-base=4.4.1-1.2204.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy R packages from all potential library paths
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder /usr/lib/R/site-library /usr/lib/R/site-library
COPY --from=builder /usr/lib/R/library /usr/lib/R/library

# Verify R installations in runtime
RUN Rscript -e 'library(EpiDISH); library(sva); library(pheatmap); library(ExperimentHub); library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)'


# Copy application files
COPY . .

# Verify fa2 imported correctly in runtime
RUN python3 -c "import fa2"

# Create /work directory for volume mount
RUN mkdir -p /work
WORKDIR /work

# Rely on the pip-installed tecpg package in dist-packages; dropping PYTHONPATH=/app to avoid shadowing

# Entrypoint setup
CMD ["python3", "-m", "tecpg"]
