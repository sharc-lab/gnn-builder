.. toctree::
   :hidden:
   :maxdepth: 2
   
   setup
   overview
   simple-tutorial

==========
GNNBuilder
==========

GNNBuilder is a framework for generating FPGA hardware accelerators for graph neural networks (GNNs) using High-Level Synthesis (HLS).

GNNBuilder is developed and maintained by the `Stefan Abi-Karam <https://stefanabikaram.com/>`_ from `Sharc Lab <https://sharclab.ece.gatech.edu/>`_ at `Georgia Tech <https://www.gatech.edu/>`_.

Quick Guide
===========

See the sidebar for links to different sections of the documentation.

* :doc:`setup` - Instructions for installing and setting up GNNBuilder.
* :doc:`overview` - Overview of the GNNBuilder framework explaining the different components and how they work together.
* :doc:`simple-tutorial` - A simple tutorial for using GNNBuilder to generate a complete GNN accelerator from start to finish including testbench generation, testbench evaluation, HLS Synthesis, IP export for Vivado, IP export for Vitis, bitstream generation for the Vitis flow, and on-device execution of the GNN accelerator using the Vitis flow and XRT runtime.


Source Code
===========

The source code repository is hosted on GitHub under our Sharc Lab organization:

* `https://github.com/sharc-lab/gnn-builder <https://github.com/sharc-lab/gnn-builder>`_


Publications
============

GNNBuilder has been published in the following places:

* FPL 2023:
   * Slides: :download:`PDF File <slides/[FPL2023] GNNBuilder - Stefan Abi-Karam.pdf>`
   * Paper: 
* ArXiv: `https://arxiv.org/abs/2303.16459 <https://arxiv.org/abs/2303.16459>`_
* WDDSA 2022 (MICRO Workshop): `https://www.escalab.org/wddsa2022/ <https://www.escalab.org/wddsa2022/>`_



Citing and Referencing
======================

If you use GNNBuilder in your research, please cite the primary FPL 2023 conference paper:

.. code-block:: text
   :class: wrap

   S. Abi-Karam and C. Hao, "GNNBuilder: An Automated Framework for Generic Graph Neural Network Accelerator Generation, Simulation, and Optimization," in 2023 33nd International Conference on Field-Programmable Logic and Applications (FPL), Gothenburg, Sweden: IEEE, Sep. 2023.

.. code-block:: bibtex

   @inproceedings{abi-karam_gnnbuilder_2023,
      location = {Gothenburg, Sweden},
      title = {{GNNBuilder}: An Automated Framework for Generic Graph Neural Network Accelerator Generation, Simulation, and Optimization},
      eventtitle = {2023 33nd International Conference on Field-Programmable Logic and Applications ({FPL})},
      booktitle = {2023 33nd International Conference on Field-Programmable Logic and Applications ({FPL})},
      publisher = {{IEEE}},
      author = {Abi-Karam, Stefan and Hao, Cong},
      date = {2023-09},
   }


.. Additional Navigation
.. =====================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`