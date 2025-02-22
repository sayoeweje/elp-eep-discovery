# Self-Assembling Protein Nanoparticles for Cytosolic Delivery of Therapeutic Macromolecules

## Description

This repository contains the necessary script and associated data to conduct the alpha helical peptide database screen performed to identify the endosomal escape peptides (EEPs) described in the paper "Self-Assembling Protein Nanoparticles for Cytosolic Delivery of Therapeutic Macromolecules".

## Contents

### Scripts
* elp-eep_discovery.py: Python file containing the full analysis pipeline

### Data
* CPPTrainingData.xlsx: Dataset used for training the first generation linear regression model. Data was derived from "Guay, D., DEL’GUIDICE, T. & LEPETIT-STOFFAES, J.-P. Polypeptide-based shuttle agents for improving the transduction efficiency of polypeptide cargos to the cytosol of target eukaryotic cells, uses thereof, methods and kits relating to same. (2016)"
* AHDB.xlsx: Alpha helical peptide database compiled from (1) the [Antimicrobial Peptide Database (APD)](https://aps.unmc.edu/), (2) [Database of Antimicrobial Activity and Structure of Peptides (DBAASP)](http://dbaasp.org/home), (3) [Therapeutic Peptide Design database (TP-DB)](https://www.nature.com/articles/s41467-021-27655-0). Filtering criteria for selection of peptides from these databases for inclusion into the AHDB is described in the methods of the paper.
* ELP-EEPs_siGFP_screen_Gen2model.xlsx: Dataset used for training the second generation linear regression model.
* ahdb_dimers_uniquepeptides_eff.xlsx: 134 dimeric alpha helical peptide designs predicted to meet efficacy criteria used for the second generation screen. No peptide monomer appears more than once in the set. Selection criteria described in the paper (Supplemental Figure 5c)
* ahdb_dimers_uniquepeptides_eff_centroids.xlsx: 32 centroids selected from 134 peptides predicted to meet efficacy criteria used for the second generation screen. Centroids were determined via [Butina clustering](https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html), using pairwise sequence alignment scores as a distance metric.

## Installation

The script requires Python 3.7 or above.

### Install dependencies
```
pip install biopython==1.78
```

### Clone Repository
```
git clone https://github.com/sayoeweje/elp-eep-discovery.git
cd elp-eep-discovery
```

### Run script
```
python /scripts/elp_eep_discovery.py
```

### Expected output
* Coefficients and loss (RMSE) for first generation linear regression model
* Coefficients and loss (RMSE) for second generation linear regression model

## License

This repository is shared under the MIT License.

