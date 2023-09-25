
<div>
    <h1 align="center">🛠️ GNNBuilder</h1>
    <p align="center">
    A framework for generating FPGA hardware accelerators for graph neural networks (GNNs) using High Level Synthesis (HLS).
    </p>
    <p align="center">
    GNNBuilder is developed and maintained by the <a href="https://stefanabikaram.com/">Stefan Abi-Karam</a> from <a href="https://sharclab.ece.gatech.edu/">Sharc Lab</a> at <a href="https://www.gatech.edu/">Georgia Tech</a>.
    </p>
</div>

## Demonstration of GNNBuilder

Below is a demo of how to use GNNBUilder to generate an FPGA accelerator for Graph-Level Classification.

```python
import gnnbuilder as gnnb

# TODO: Add example code
```

A user can provide a trained PyTorch model, and the GNNBuilder framework can generate high-level synthesis (HLS) C++ code for the model.

This HLS model can then be used to build a bitstream of the model that can be executed on an FPGA.

This framework also provides some simple design space exploration (DSE) tools using a performance model that the designer can use to parameterize the hardware implementation in terms of parallelism and numerical precision for accuracy, latency, and resource usage tradeoffs.

## Documentation

Comprehensive documentation is available at the following link: TBD.

You can find information about package installation, setup, demo usage, common use cases, and package API details.

## Quick Installation

We have packaged GNNBuilder as both a `pip` package and a `conda` package to streamline the installation process.

To install GNNBuilder using `pip`, run the following command:

```bash
pip install git+https://github.com/sharc-lab/gnn-builder.git​
```

To install GNNBuilder using `conda` (or equivalent package manager like `mamba`), run the following command:

```bash
conda install gnnbuilder
```

## Requirements and System Setup

The core library itself only requires Python packages and does not require any external tools.

However, without external tools, the library will only be able to generate source code and build scripts for the user's design.

To build the design, testbenches, and hardware bitstreams, the user will need to have installed the Xilinx Vitis HLS, Vivado, and Vitis tools on their system in additon to some standard Linux tools like `make` and `clang`.

For ease of setup, we highly recommend using a Linux-based operating system. We have only really tested GNNBuilder on Linux systems.

We are also currently working on composing all the necessary tools and libraries into a Docker image for ease of use across different systems.

## Citing and Referencing GNNBuilder

You can cite GNNBuilder as it was mainly published at FPL 2023.

```text
S. Abi-Karam and C. Hao, "GNNBuilder: An Automated Framework for Generic Graph Neural Network Accelerator Generation, Simulation, and Optimization," in 2023 33nd International Conference on Field-Programmable Logic and Applications (FPL), Gothenburg, Sweden: IEEE, Sep. 2023.
```

```bibtex
@inproceedings{abi-karam_gnnbuilder_2023,
    location = {Gothenburg, Sweden},
    title = {{GNNBuilder}: An Automated Framework for Generic Graph Neural Network Accelerator Generation, Simulation, and Optimization},
    eventtitle = {2023 33nd International Conference on Field-Programmable Logic and Applications ({FPL})},
    booktitle = {2023 33nd International Conference on Field-Programmable Logic and Applications ({FPL})},
    publisher = {{IEEE}},
    author = {Abi-Karam, Stefan and Hao, Cong},
    date = {2023-09},
}
```

## Associated Publications

- FPL 2023: Link comming soon...
- ArXiv: [https://arxiv.org/abs/2303.16459](https://arxiv.org/abs/2303.16459)
- WDDSA 2022: [https://www.escalab.org/wddsa2022/](https://www.escalab.org/wddsa2022/)
