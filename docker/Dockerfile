# Use the official Julia image with the desired version
FROM julia:1.10.5

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the Project.toml and Manifest.toml files from the root of the project
COPY ../Project.toml ../Manifest.toml ./

# Install necessary system dependencies for JupyterLab and VSCode integration
RUN apt-get update && \
    apt-get install -y python3-pip curl && \
    pip3 install jupyterlab

# Install Julia packages based on the Project.toml file
RUN julia -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

# Install Julia extension for VSCode support
RUN julia -e 'using Pkg; Pkg.add("LanguageServer"); Pkg.add("SymbolServer")'

# Expose the Jupyter notebook port and VS Code extension support
EXPOSE 8888 9000

# Start Jupyter Lab when the container starts
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
