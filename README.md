# Introduction
This repository contains Julia code for research in industrial statistics.

See the [wiki](https://github.com/ben-n-fuller/industrial-stats/wiki) for further documentation.

# Usage
The notebooks and code in this repo can be run through VS Code or in Jupyter Lab.

## Install Julia
Julia installation and version management is best handled by the `juliaup` package. Instructions for different platforms are available in the [official repo](https://github.com/JuliaLang/juliaup).

Then, in the project root, install the required packages.

```
julia -e "import Pkg; Pkg.activate()"
```

## Install Python (Optional)
Python installation and version management is best handled by `miniconda`. Instructions for different platforms are available in the [official documentation](https://docs.anaconda.com/miniconda/).

## Install Docker (Optional)
To use Jupyter Lab with this project, either Linux or WSL are required.

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) (or [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) if running in WSL).

## VS Code Setup
1. Install the official `Jupyter` extension
2. Install the official `Julia` language extension
3. Open a notebook and select the kernel in the top right corresponding with the `juliaup` version just installed

## Jupyter Lab Setup
1. In the `docker/` directory run `docker compose build`
2. In the same directory run `docker compose up -d`
3. Navigate to `localhost:8888/lab` in the browser to edit and run code


