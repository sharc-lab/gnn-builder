# 🛠️ GNNBuilder 

GNNBuilder is a framework for generating FPGA hardware accelerators for graph neural networks (GNNs).

It is developed and maintained by the [Stefan Abi-Karam](https://stefanabikaram.com/) from [Sharc Lab](https://sharclab.ece.gatech.edu/) at [Georgia Tech](https://www.gatech.edu/).


## Initial Demo

Below is a simple demo of the GNNBuilder framework from `demos/demo.py`.

```python
import gnnbuilder as gnnb

# TODO: Add example code
```

A user can provide a trained PyTorch model, and the GNNBuilder framework can generate high-level synthesis (HLS) C++ code for the model.

This HLS model can then be used to build a bitstream of the model that can be executed on an FPGA. 

This framework also provides some simple design space exploration (DSE) tools using a performance model that the designer can use to parameterize the hardware implementation in terms of parallelism and numerical precision for accuracy, latency, and resource usage tradeoffs.

## Requirements

For ease of setup, we highly recommend using a Linux-based operating system. The system must be capable of running the Xilinx FPGA tools.
We are currently working on composing all the necessary tools and libraries into a Docker image for ease of use.

GNNBuilder requires the following Python packages:

- PyTorch
- PyTorch Geometric
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jinja2
- tqdm
- Joblib

GNNBuilder also requires the following external tools:

- Xilinx Vitis HLS
- Xilinx Vivado
- Xilinx Vitis
- Xilinx XRT Library (only needed for running on an FPGA)
- Clang or GCC
- ld
- Make

## Installation and Configuration

## Documentation

## More Examples

## Referencing GNNBuilder

## Associated Publications
