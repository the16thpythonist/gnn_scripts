|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/

=============
GNN Classification Scripts
=============

Some scripts to train various graph neural networks (GNNs) to solve a binary classification task.

Installation
============

1. Clone the repository

.. code-block:: shell

    git clone https://github.com/the16thpythonist/gnn_scripts.git

2. (Optional) Create and activate a virtual environment

.. code-block:: shell

    cd gnn_scripts
    python3 -m venv venv
    source ./venv/bin/activate

3. Then in the main folder run a ``pip install`` with the editable flag ``-e`` - this will ensure that 
changes in the code will be reflected when executing the scripts:

.. code-block:: shell

    cd gnn_scripts
    python3 -m pip install -e .


Usage
=====

Computational Experiments
-------------------------

The various models can be trained by executing the experiment scripts in the ``gnn_scripts/experiments`` folder. That folder is used to 
collect all the executable scripts, while the rest of the package contains functions for utility, visualization etc.

The experiment scripts are built on the PyComex_ library for the simplified managment of computational experiments. By structuring each module 
in a specific manner predetermined by the library using the ``Experiment`` construct, the library will handle various tasks related to 
the execution of the computational experiments automatically. Most importantly, for each independent run a new archival folder containing 
all the evaluation artifacts will be created automatically in the ``gnn_scripts/experimetns/results`` folder.

Each experiment can simply be executed by

.. code-block: shell
    
    # example
    python3 train_model__gcn.py


Model Training Experiments
--------------------------

- ``train_model.py``: This is is the *base* experiment implementation, which means it contains all the code that all of the various experiments 
  have in common. Executing this script will not have a result since it does not contain any concrete model implementation, it merely defines 
  how the dataset is loaded and pre-processed, as well as the implementation for the visualization of the classification results. 
  All the subsequent experiments extend on this module.
- ``train_model__gcn.py``: Trains a GCN-based network to solve a binary classification task. 
- ``train_model__gat.py``: Trains a GATv2-based network to solve a binary classification task.

SMILES Datasets
---------------

All the experiment scripts are exectuble by themselves, but will only use the ``test.csv`` dataset which contains randomly generated class labels 
for testing purposes.

To use the scripts it will be necessary to define a custom dataset by supplying the absolute path in the ``CSV_PATH`` global variable. Additionally 
the column names of the smiles column and the target value column have to be adjusted in the ``SMILES_COLUMN_NAME`` and ``TARGET_COLUMN_NAME`` variables.

Credits
=======

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git