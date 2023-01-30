########################
Online Tutorial
########################
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/pitt-miskov-zivanov-lab/CLARINET/HEAD?labpath=%2Fexamples%2Fuse_CLARINET.ipynb

Click the icon above and run the demonstrated example; or alternatively upload user-customized input files to the ``input`` directory on :guilabel:`File Browser` tab (upper left corner) of Binder.

This interactive jupyter notebook walks you though all of the code and functions to:
  * Get familiar with and parse the input files including baseline model spreadsheet and machine reading extracted events.
  * Create ECLG, process it using network algorithm and assign weights.
  * Cluster the ECLG using the community detection algorithm and possibly merge clusters.

########################
Offline Installation
########################

CLARINET requires Python installed on local machine (MacOS, Linux and Windows are all supported). If users want to explore the interactive notebook we provided locally, Jupyter notebook installation is also required.

1. Clone the `CLARINET repository <https://github.com/pitt-miskov-zivanov-lab/CLARINET>`_, to your computer.

.. code-block:: bash

   git clone https://github.com/pitt-miskov-zivanov-lab/CLARINET.git

2. Navigate into the directory, install CLARINET and its python dependencies.

.. code-block:: bash

   cd CLARINET
   pip install -e .

3. Run the provided notebook.

.. code-block:: bash

  jupyter notebook examples/use_CLARINET.ipynb

########################
Input and Output
########################

Input includes:
  * a .xlsx file containing the model to extend, in the BioRECIPES tabular format, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/input/BooleanTcell.xlsx>`_
  * a machine reading output file with the following header, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/input/MachineReadingOutput.csv>`_ |br| RegulatedName, RegulatedID, RegulatedType, RegulatorName, RegulatorID, RegulatorType, PaperID
  * number of return paths
  * a parameter for frequency class

Output includes:
  * a frequent class file that contains ECLG nodes and their freqClass level, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/freqClass>`_
  * the ECLG file containing nodes and edges before the removal of less frequent nodes, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/ECLGbefore.txt>`_
  * the ECLG file containing nodes and edges after the removal of less frequent nodes, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/ECLGafter.txt>`_
  * a LSS file that contains ECLG edges and their weights, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/LSS>`_
  * a pickle file containing grouped (clustered) extensions, specified as nested lists. Each group starts with an integer, followed by interactions specified as [regulator element, regulated element, Interaction type: Activation (+) or Inhibition (-)], `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/grouped_ext>`_
  * another pickle file containing the merged clusters (different than _grouped_ext_ which is not merged), clusters are merged based on user-selected number of return paths, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/grouped_ext_Merged>`_
  * directory containing the resulting uninterpreted clusters, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/tree/main/examples/output/GeneratedClusters>`_
  * a .csv file showing the basic information of each uninterpreted cluster, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/blob/main/examples/output/ClusterInfoFile.csv>`_
  * directory containing the resulting interpreted clusters, `see example <https://github.com/pitt-miskov-zivanov-lab/CLARINET/tree/main/examples/output/InterpretedClusters>`_

########################
Dependency Resources
########################

  * `NetworkX - Network Analysis in Python  <https://networkx.org>`_, being used in many core functions in CLARINET
  * `python-louvain - Community Detection <https://pypi.org/project/python-louvain/>`_, being used to cluster the ECLG into communities (clusters)

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />
