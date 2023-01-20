.. CLARINET documentation master file, created by
   sphinx-quickstart on Wed May 19 13:37:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CLARINET's documentation!
====================================
CLARINET (CLARIfying NETworks) ia a novel tool for rapid model assembly by automatically extending dynamic network models with the information published in literature. This facilitates information reuse and data reproducibility and replaces hundreds or thousands of manual experiments, thereby reducing the time needed for the advancement of knowledge.  

CLARINET objectives
-------------------
1.	Utilizing the knowledge published in literature and suggests model extensions.
2.	Studying events extracted from literature as a collaboration graph, including several metrics that rely on the event occurrence and co-occurrence frequency in literature.
3.	Allowing users to explore different selection criteria when automatically finding best extensions for their models.

CLARINET architecture
---------------------
(Left) CLARINET inputs: Extracted Event Set (EES) and Baseline model. (Right) Flow diagram of the CLARINET processing steps and outputs.

.. image:: CLARINET1.png



Dependencies
------------
Python libraries: pandas, numpy, network, math, pickle, community, matplotlib.pyplot. 

Applications
------------
The primary application area of CLARINET is dynamic and causal network models.

Funding
-------
CLARINET was partially supported by the AIMCancer DARPA award (W911NF-17-1-0135).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CLARINET
   Legal
 



