==================
Framework Overview
==================

GNNBuilder has several components that work together in a larger, framework-defined workflow. Both the components and the workflow are exposed to the user via the GNNBuilder library. This enables users to create custom workflows or integrate GNNBuilder into their current experiments and design flows for their downstream applications of interest.


Workflow
========

It's easier to understand GNNBuilder from a workflow perspective to see how one uses the framework to implement a hardware accelerator for a GNN model.

.. figure:: /figures/gnnbuilder-overview.png
   :width: 100%
   :align: center
   :alt: GNNBuilder Overview

   Diagram of the GNNBuilder workflow.

The figure above is a comprehensive overview of the GNNBuilder workflow. Key components are described below in more detail.

Frontend API
------------

The frontend API exposes the GNNBuilder framework and its components to the user via the Python library.

One major component of the API is the components that it exposes that allow users to build native Pytorech Geornmic GNN modules that are compatible with the GNNBuilder framework. This includes components for Graph Convolutional Layers, Global Graph Pooling Layers, Aggragions, and MLP heads for node, edge, and graph level task readout. We also provide a simple GNN ``Model`` class that parameterizes a simple GNN model architecture that supports node, edge, and graph level readout with skip connections, custom layer sizes, custom layer types, global graph pooling, and  MLP head. We see that this model architecture covers most use cases for GNNs or at least the core compute performed on graphs.

The frontend API has modules and classes that allow the user to perform all the build and automation tasks outlined in the workflow above. This includes generating source code and build files in an organized project directory, calling Vitis HLS to run HLS synthesis, and calling Vitis to compile and implement the final hardware bitstream and deployment when targeting Vitis-supported platforms.

Compiler and Code Generator
---------------------------

The GNNBuilder compiler and code generator are the core of the GNNBuilder framework. Put simply, GNNBuilder is a template-based compiler that lowers a high-level representation of a GNN model and components down to HLS C++ source code.

A user-defined PyTorch GNN model is decomposed into its parts and their associated configurations.


Testbench Simulation
--------------------

The GNNBuilder framwork is able to genrate C++ HLS testbenches that can call the generated HLS kernel with data serilzied form the Pytoch code. This includes both the weights from the defined PyTorch Gemoirtic GNN model as well as input graphs and features from the user's dataset. This data is formatted and serialized to disk in a way that makes it simple for the generated C++ testbench to then load the data from disk and pass it into the HLS kernel.

The generated testbench code also records the output of the HLS kernel and computes both the task loss/accuracy as well as the expected PyTorch output vs. the C++ Kernel output. These metrics are also recorded to disk and are parsed by the Python frontend API to provide the user with a simple way to compare the PyTorch and HLS kernel outputs when running the testbench via the Python frontend API.

The generated testbench should also be compatible with the Vitis HLS Co-Simulation feature which is experimental at the moment. More updates on this will be provided in the future.


Synthesis and Deployment
------------------------

The GNNBuilder framework provides the option to Vitis HLS to synthesize the generated HLS kernel using the Python frontend API. The reported results from synthesis are also parsed by the framework and are provided to the user when synthesis is called via the Python frontend API. This data includes latency and resource utilization metrics.

Experimental support for the Vitis flow is also provided. This includes the ability to generate a Vitis project and call the Vitis compiler to compile and implement the final hardware bitstream. The user can also deploy the bitstream to a Vitis-supported platform such as the Xilinx Alveo U250 or U280 accelerator cards. This also includes the generation of C++ OpenCL host code that can be used to run the generated bitstream on the target platform.

Design Space Exploration Performance Models
-------------------------------------------

Two simple DSE models are provided that can be used to estimate the latency and BRAM usage of a given user-defined model configuration. These models are used to provide the user with a quick estimate of the performance and resource usage without having to run HLS synthesis. This is useful for quickly exploring the design space of a given model and its configurations.