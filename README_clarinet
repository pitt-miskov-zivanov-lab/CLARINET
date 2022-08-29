CLARINET

-CLARIfying NETworks

An efficient approach for informing dynamic network models with knowledge from literature



-Important abbreviations 

ECLG --> Event ColLaboration Graph
FC_IA --> Individual assessment of events using Frequency Class concept (node weights)
FC_PA --> Pair assessment of events using Frequency Class concept (edge weights)
IF_PA --> Pair assessment of events using Inverse Frequency concept (edge weights)



-Functionality

An automated framework for rapid model assembly by automatically extending models with the information published in literature. This facilitates information reuse and data reproducibility and replaces hundreds or thousands of manual experiments, thereby reducing the time needed for the advancement of knowledge. 


-Description of files

.runClarinet.py functions for extending discrete network models in the BioRECIPES tabular format using knowledge from literature. The output is a set of node-edge weighted graph clusters.
.runClusterInfo functions for ranking the generated clusters 
.BooleanTcell.xlsx example of a baseline model in the BioRECIPES format
.MachineReadingOutput.csv example of machine reading output
.TCELLgraph.graphml the graphml representation of the baseline model 


-I/O

Input

.Machine Reading output file with the following header
RegulatedName,RegulatedID,RegulatedType,RegulatorName,RegulatorID,RegulatorType,PaperID
.A file containing the model to extend in the BioRECIPES tabular format
.The number of return paths


Output

.grouped_ext A pickle file containing grouped (clustered) extensions, specified as nested lists. Each group starts with an integer, followed by interactions specified as [regulator element, regulated element, Interaction type: Activation (+) or Inhibition (-)
This file along with the directory of system properties will be the input to the statistical model checking to verify the behavior of candidate models against the properties
.grouped_ext_Merged A pickle file containing the merged clusters
.GeneratedCluster Directory that contains the generated clusters each cluster in a separate file
.InterpretedClusters Directory that contains the interpreted clusters that correspond to the generated clusters

