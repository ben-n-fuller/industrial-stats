# Use the official Julia image with the desired version
FROM julia:1.10.5

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

# Add Conda to the path
ENV PATH="/root/miniconda3/bin:$PATH"

# Initialize conda for bash
RUN ~/miniconda3/bin/conda init bash

# Install jupyter lab
RUN conda install -y jupyterlab

# Copy the Project.toml and Manifest.toml files from the root of the project
COPY ../Project.toml ../Manifest.toml ./

# Install Julia packages based on the Project.toml file
# RUN julia -e 'import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'

# Install and configure the Julia kernel for JupyterLab
RUN julia -e 'import Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("Julia", "--project=@.")'

# Set the JULIA_PROJECT environment variable to point to the current environment
ENV JULIA_PROJECT=@.

# Install and configure the Julia kernel for JupyterLab
# RUN julia -e 'import Pkg; Pkg.add("IJulia"); using IJulia; notebook(detached=true)'

# Install Julia extension for VSCode support
# RUN julia -e 'using Pkg; Pkg.add("LanguageServer"); Pkg.add("SymbolServer")'

# Expose the Jupyter notebook port and VS Code extension support
EXPOSE 8888 9000

# Start Jupyter Lab when the container starts
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
