# CLARINET
[![Documentation Status](https://readthedocs.org/projects/melody-clarinet/badge/?version=latest)](https://melody-clarinet.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pitt-miskov-zivanov-lab/CLARINET/HEAD?labpath=%2Fexamples%2Fuse_CLARINET.ipynb)

### (CLARIfying NETworks)

CLARINET (CLARIfying NETworks) ia a novel tool for rapid model assembly by automatically extending dynamic network models with the information published in literature. This facilitates information reuse and data reproducibility and replaces hundreds or thousands of manual experiments, thereby reducing the time needed for the advancement of knowledge.

## Contents

- [Functionality](#Functionality)
- [Important Abbreviations](#Important-Abbreviations)
- [I/O](#IO)
- [Online Tutorial](#Online-Tutorial)
- [Offline Installation](#Offline-Installation)
- [Package Structure](#Package-Structure)
- [Citation](#Citation)
- [Funding](#Funding)
- [Support](#Support)

## Functionality
- Extraction: utilizing the knowledge published in literature and suggests model extensions, studying events extracted from literature as a collaboration graph
- Weighting: assigning weights to collaboration graph using a variety of network metrics
- Clustering: detecting communities within the collaboration graph, creating groups of interactions

## Important Abbreviations

- **ECLG**: Event ColLaboration Graph
- **FC_IA**: Individual assessment of events using Frequency Class concept (node weights)
- **FC_PA**: Pair assessment of events using Frequency Class concept (edge weights)
- **IF_PA**: Pair assessment of events using Inverse Frequency concept (edge weights)

## I/O

### Input
- A .xlsx file containing the model to extend, in [BioRECIPES model](https://melody-biorecipe.readthedocs.io/en/latest/model_representation.html) format, see [`examples/input/BooleanTcell_biorecipe.xlsx`](examples/input/BooleanTcell_biorecipe.xlsx)
- Machine reading output file, in [BioRECIPES interaction](https://melody-biorecipe.readthedocs.io/en/latest/bio_interactions.html) format, see [`examples/input/ReadingOutput_biorecipe.csv`](examples/input/ReadingOutput_biorecipe.csv)
- Parameter for frequency class weighting, defined in Cell 14 of the notebook
- Number of return paths, defined in Cell 22 of the notebook

### Output

- [`examples/output/freqClass`](examples/output/freqClass), a frequent class file that contains ECLG nodes and their freqClass level
- intermediate output - [`examples/output/ECLGbefore.txt`](examples/output/ECLGbefore.txt), the ECLG file containing nodes and edges before the removal of less frequent nodes
- intermediate output - [`examples/output/ECLGafter.txt`](examples/output/ECLGafter.txt), the ECLG file containing nodes and edges after the removal of less frequent nodes
- [`examples/output/LSS`](examples/output/LSS), a LSS file that contains ECLG edges and their weights
- [`examples/output/grouped_ext`](examples/output/grouped_ext), a pickle file containing grouped (clustered) extensions, specified as nested lists. Each group starts with an integer, followed by interactions specified as [regulator element, regulated element, Interaction type: Activation (+) or Inhibition (-)]
- [`examples/output/grouped_ext_Merged`](examples/output/grouped_ext_Merged), another pickle file containing the merged clusters (different than _grouped_ext_ which is not merged), clusters are merged based on user-selected number of return paths
- [`examples/output/GeneratedClusters`](examples/output/GeneratedClusters), directory containing the resulting uninterpreted clusters
- intermediate output - [`examples/output/ClusterInfoFile.csv`](examples/output/ClusterInfoFile.csv), a .csv file showing the basic information of each uninterpreted cluster
- [`examples/output/InterpretedClusters`](examples/output/InterpretedClusters), directory containing the resulting interpreted clusters

## Online Tutorial
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pitt-miskov-zivanov-lab/CLARINET/HEAD?labpath=%2Fexamples%2Fuse_CLARINET.ipynb)

Run the demonstrated example; or alternatively upload user-customized input files (see [I/O](#IO)) to the _input/_ directory on File Browser Tab (upper left corner) of Binder.

#### This interactive jupyter notebook walks you though all of the code and functions to:

1. Get familiar with and parse the input files including baseline model spreadsheet and machine reading extracted events.
2. Create ECLG, process it using network algorithm and assign weights.
3. Cluster the ECLG using the community detection algorithm and possibly merge clusters.

## Offline Installation

1. Clone the CLARINET repository to your computer.
   ```
   git clone https://github.com/pitt-miskov-zivanov-lab/CLARINET.git
   ```
2. Navigate into the directory, install CLARINET and its python dependencies.
   ```
   cd CLARINET
   pip install -e .
   ```
3. Run the provided notebook (Check [Jupyter notebook installation](https://jupyter.org/install) here).
   ```
   jupyter notebook examples/use_CLARINET.ipynb
   ```

## Package Structure

- [`setup.py`](setup.py): python file that help set up python dependencies installtion
- [`src/`](src/): directory that includes core python CLARINET files
  - [`src/runClarinet.py`](src/runClarinet.py): functions for extending discrete network models in the [BioRECIPE](https://melody-biorecipe.readthedocs.io) format using knowledge from literature
- [`examples/`](examples/): directory that includes tutorial notebook and example inputs and outputs
- [`environment.yml`](environment.yml): environment file, required by [Binder](https://mybinder.readthedocs.io/en/latest/using/config_files.html#environment-yml-install-a-conda-environment)
- [`docs/`](docs/): containing files supporting the repo's host on [Read the Docs](https://theclarinet.readthedocs.io)
- [`supplementary/`](supplementary): containing supplementary files for the studied model
  - [`supplementary/TCELLgraph.graphml`](supplementary/TCELLgraph.graphml): the graphml representation of the baseline model
- [`LICENSE.txt`](LICENSE.txt): MIT License
- [`README.md`](README.md): this is me!

## Citation

_Yasmine Ahmed, Cheryl A Telmer, Natasa Miskov-Zivanov, CLARINET: efficient learning of dynamic network models from literature, Bioinformatics Advances, Volume 1, Issue 1, 2021, vbab006, https://doi.org/10.1093/bioadv/vbab006_

## Funding

This work was funded in part by DARPA Big Mechanism award, AIMCancer (W911NF-17-1-0135); and in part by the University of Pittsburgh, Swanson School of Engineering.

## Support
Feel free to reach out via email nmzivanov@pitt.edu for additional support if you run into any error.
