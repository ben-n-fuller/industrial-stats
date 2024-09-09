# Introduction
This repository contains Julia code for computational research in industrial statistics.

- `/src` contains a library of modules defining common operations
- `/notebooks` contains exploratory Jupyter notebooks
- `/examples` contains examples of code use and common workflows

See the [wiki](https://github.com/ben-n-fuller/industrial-stats/wiki) for further documentation.

# Usage
The notebooks and code in this repo can be run through VS Code or in Jupyter Lab.

## Install Julia
If Julia is not already installed, `juliaup` can be very useful for installing and managing Julia versions. Instructions for different platforms are available in the [official repo](https://github.com/JuliaLang/juliaup).

Once installed, activate the local package environment in the root directory:

```
julia -e "import Pkg; Pkg.activate(\".\"); Pkg.instantiate()"
```

## Install Conda
The `conda` environment manager is used when installing `IJulia` for Julia-based Jupyter notebooks. Instructions for different platforms are available in the [official documentation](https://docs.anaconda.com/miniconda/).

## VS Code Setup
1. Install the official `Jupyter` extension
2. Install the official `Julia` language extension
3. Run `julia -e "using IJulia; IJulia.notebook(detached=true)"` in the command line to start the Jupyter server
4. Open the root folder in vscode, open a notebook, and select the kernel in the top right

## Jupyter Lab Setup (optional)
1. In the `docker/` directory run `docker compose build`
2. In the same directory run `docker compose up -d`
3. Navigate to `localhost:8888/lab` in the browser to edit and run code

## Install Docker (Optional)
To use Jupyter Lab with this project, either Linux or WSL are required.

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) (or [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) if running in WSL).
