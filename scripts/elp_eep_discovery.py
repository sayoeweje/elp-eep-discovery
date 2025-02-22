"""# Packages"""

import pandas as pd
import numpy as np
from numpy import genfromtxt
import scipy as sp
from matplotlib import pyplot as plt
import math

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
from Bio.Align import substitution_matrices
from Bio.SubsMat import MatrixInfo as matlist

from rdkit.ML.Cluster import Butina

"""# Functions"""

hscale_wif = {'G': 0.01, 'A': 0.17, 'V': 0.07, 'L': -0.56, 'I': -0.31, 'M': -0.23,
              'P': 0.45, 'F': -1.13, 'W': -1.85, 'S': 0.13, 'T': 0.14, 'N': 0.42,
              'Q': 0.58, 'Y': -0.94, 'C': -0.24, 'K': 0.99, 'R': 0.81, 'H': 0.96,
              'D': 1.23, 'E': 2.02}

pKa = {'A':  [2.34, 9.69], 'R': [2.17, 9.04, 12.48], 'N': [2.02, 8.8],
       'D': [2.19, 9.67, 4.25], 'C':  [1.9, 8.18, 5.07], 'Q': [2.17, 9.13],
       'E': [2.19, 9.67, 4.25], 'G':  [2.34, 9.6], 'H': [1.82, 9.17, 6.00],
       'I':  [2.36, 9.6], 'L':  [2.36, 9.6], 'K': [2.18, 8.95, 10.53],
       'M':  [2.28, 9.21], 'F':  [1.83, 9.13], 'P': [1.99, 10.60],
       'S': [2.21, 9.15], 'T': [2.09, 9.10], 'W':  [2.83, 9.39],
       'Y':  [2.20, 9.11], 'V':  [2.32, 9.62]}

def calculate_hydrophobic_moment(peptide):
    phi = 0
    theta = 100/180*math.pi
    x = 0
    y = 0
    for residue in peptide:
        x = x - hscale_wif[residue]*math.sin(phi)
        y = y + hscale_wif[residue]*math.cos(phi)
        phi = phi + theta
    return math.sqrt(x**2+y**2)

def calculate_pI(peptide):
    first_res = peptide[0]
    last_res = peptide[-1]
    pKa_array = [pKa[first_res][1], pKa[last_res][0]]
    for residue in peptide:
        charged_res = ['R', 'K', 'D', 'E', 'H']
        if residue in charged_res:
            pKa_array.append(pKa[residue][2])
    pKa_array.sort()
    pepcharge = peptide.count('R')+peptide.count('K')+peptide.count('H')+1
    pI = (pKa_array[pepcharge-1]+pKa_array[pepcharge])/2
    return pI

def calculate_regression1_features(df):
    for index, row in df.iterrows():
        peptide = row["Sequence"]
        df.loc[index, "Amphipathic alpha-helix"] = 1
        df.loc[index, "Hydrophobic moment"] = calculate_hydrophobic_moment(peptide)
        df.loc[index, "% Negative residues"] = (peptide.count("D")+peptide.count("E"))/len(peptide)*100
        df.loc[index, "Net charge"] = peptide.count("K")+peptide.count("R")-peptide.count("D")-peptide.count("E")
        df.loc[index, "pI"] = calculate_pI(peptide)
        df.loc[index, "Predicted Efficacy"] = 5.90*df.loc[index, 'Hydrophobic moment'] + 3.19*df.loc[index, 'Net charge'] - 3.23*df.loc[index, 'pI']- 4.34*df.loc[index, '% Negative residues'] + 20.77*df.loc[index, "Amphipathic alpha-helix"]
    return df

def calculate_regression2_features(df):
    for index, row in df.iterrows():
        peptide = row["Sequence"]
        df.loc[index, "Amphipathic alpha-helix"] = 1
        df.loc[index, "Hydrophobic moment"] = calculate_hydrophobic_moment(peptide)
        df.loc[index, "% Hydrophobic residues"] = (peptide.count("A")+peptide.count("C")+peptide.count("G")+peptide.count("I")+peptide.count("L")+peptide.count("M")+peptide.count("F")+peptide.count("P")+peptide.count("W")+peptide.count("Y")+peptide.count("V"))/len(peptide)*100
        df.loc[index, "Length"] = len(peptide)
        df.loc[index, "% Negative residues"] = (peptide.count("D")+peptide.count("E"))/len(peptide)*100
        df.loc[index, "Net charge"] = peptide.count("K")+peptide.count("R")-peptide.count("D")-peptide.count("E")
        df.loc[index, "% K of positive charges"] = peptide.count("K")/(peptide.count("K")+peptide.count("R"))*100
        df.loc[index, "% Positive residues"] = (peptide.count("K")+peptide.count("R"))/len(peptide)*100

        df.loc[index, "Predicted Efficacy"] =  1.29*df.loc[row, "% K of positive charges"] - 0.76*df.loc[row, "% Hydrophobic residues"] - 0.149*df.loc[row, "% Positive residues"]

    return df

def psa(peptide1, peptide2): # Pairwise sequence alignment
    scores = []
    alignment_score = pairwise2.align.globalds(peptide1, peptide2, matrix, -10, -10, penalize_end_gaps = False, score_only = True)
    scores.append(alignment_score)
    distance = abs(alignment_score-203) # 203 is max alignment score of two peptides from AHDB. High alignment score --> low distance --> high similarity
    return distance

def cluster_peptides(peptides, cutoff, alignmentFunc):
    clusters = Butina.ClusterData(peptides, len(peptides), cutoff, isDistData=False, distFunc=alignmentFunc)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters

"""# First generation EEPs (EEP1 - EEP8)

## Linear Regression Model Training
"""

cpp_df = pd.read_excel("data/CPPTrainingData.xlsx", index_col=0)
cpp_df = pd.get_dummies(cpp_df, columns = ["Amphipathic alpha-helix", "Positively-charged surface"], drop_first = True)

X = cpp_df.loc[:, cpp_df.columns != 'Transfection Efficiency']
Y = cpp_df.loc[:,"Transfection Efficiency"]

cv = KFold(n_splits=len(cpp_df)-1) # Leave-one-out cross validation
model = LinearRegression(fit_intercept = False)
fs = SelectKBest(score_func=f_regression) # Feature selection using F scores
pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
grid = dict()
grid['sel__k'] = [i for i in range(1, 15)]

search = GridSearchCV(pipeline, grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv)
results = search.fit(X, Y)
cols = results.best_estimator_[0].get_support(indices=True) #

# Summarize results of best model
print('Best RMSE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print('Model features: %s' % X.columns[cols])
print('Model coefficients: %s' % results.best_estimator_[1].coef_)
yhat = results.best_estimator_[1].predict(results.best_estimator_[0].transform(X)) # Model predicted transfection efficiencies

# F scores for each feature
for i in range(len(results.best_estimator_[0].scores_)):
    print(X.columns[i],': %f' % results.best_estimator_[0].scores_[i])

fig = plt.figure()

yvalues_df = pd.DataFrame(index=X.index, columns=["y", "yhat"])
yvalues_df.iloc[:, 0] = Y
yvalues_df.iloc[:, 1] = yhat

"""## AHDB selection"""

ahdb = pd.read_excel("data/AHDB.xlsx")
ahdb = calculate_regression1_features(ahdb)
ahdb_10 = ahdb[ahdb["Predicted Efficacy"] > 10]

ahdb_dimers = pd.DataFrame(columns=ahdb_10.columns, index = range(1, len(ahdb_10)**2+1))

counter = 0

for index1, row1 in ahdb_10.iterrows():
    for index2, row2 in ahdb_10.iterrows():
          sequence = ahdb_10.loc[index1, "Sequence"] + "GGSGGGS" + ahdb_10.loc[index2, "Sequence"]
          ahdb_dimers.iloc[counter, 0] = sequence
          counter += 1

ahdb_dimers = calculate_regression1_features(ahdb_dimers)
ahdb_dimers_filtered = ahdb_dimers[(ahdb_dimers["Predicted Efficacy"] > 83.7) & len(ahdb_dimers["Sequence"]) < 50] # 83.7 is highest efficacy from training data set

"""# Second generation EEPs (EEP9 - EEP15)

## Linear Regression Model Training
"""

eep_df = pd.read_excel("data/ELP-EEPs_siGFP_screen_Gen2model.xlsx", index_col=0)
eep_df = pd.get_dummies(eep_df, columns = ["Contains polyHis?", "Positively-charged surface"], drop_first = True)

X = eep_df.loc[:, eep_df.columns != 'siGFP delivery efficacy (1/(%GFP+ * [ELP] * [siRNA])']
Y = eep_df.loc[:, "siGFP delivery efficacy (1/(%GFP+ * [ELP] * [siRNA])"]

cv = KFold(n_splits=len(eep_df)-1) # Leave-one-out cross validation
model = LinearRegression(fit_intercept = False)
fs = SelectKBest(score_func=f_regression) # Feature selection using F scores
pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
grid = dict()
grid['sel__k'] = [i for i in range(1, 15)]

search = GridSearchCV(pipeline, grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv)
results = search.fit(X, Y)
cols = results.best_estimator_[0].get_support(indices=True) #

# Summarize results of best model
print('Best RMSE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print('Model features: %s' % X.columns[cols])
print('Model coefficients: %s' % results.best_estimator_[1].coef_)
yhat = results.best_estimator_[1].predict(results.best_estimator_[0].transform(X)) # Model predicted transfection efficiencies

# F scores for each feature
for i in range(len(results.best_estimator_[0].scores_)):
    print(X.columns[i],': %f' % results.best_estimator_[0].scores_[i])

fig = plt.figure()

yvalues_df = pd.DataFrame(index=X.index, columns=["y", "yhat"])
yvalues_df.iloc[:, 0] = Y
yvalues_df.iloc[:, 1] = yhat

"""## AHDB selection"""

ahdb = pd.read_excel("data/AHDB.xlsx")
ahdb_filtered = calculate_regression2_features(ahdb)
ahdb_filtered = ahdb_filtered[(ahdb_filtered['Length'] <= 20) & (ahdb_filtered['Length'] >= 10) & (ahdb_filtered["Net charge"] >= ahdb_filtered["Net charge"].mean())
                             & (ahdb_filtered['% Negative residues'] < 5) & (ahdb_filtered['Predicted Efficacy'] > 0)]

ahdb_dimers = pd.DataFrame(columns=ahdb_filtered.columns, index = range(1, len(ahdb_filtered)**2+1))
display(ahdb_dimers)

counter = 0
for index1, row1 in ahdb_filtered.iterrows():
    for index2, row2 in ahdb_filtered.iterrows():
          sequence = ahdb_filtered.loc[index1, "Sequence"] + "GGSGGGS" + ahdb_filtered.loc[index2, "Sequence"]
          ahdb_dimers.iloc[counter, 0] = sequence
          counter += 1

ahdb_dimers = calculate_regression2_features(ahdb_dimers)
ahdb_dimers_filtered = ahdb_dimers[(ahdb_dimers["Predicted Efficacy"] >= 50) & (ahdb_dimers["Hydrophobic Moment"] >= 7) & (ahdb_dimers["% Positive residues"] <= 40)]

ahdb_dimers_filtered_unique = ahdb_dimers_filtered.copy()
unique_monomers = set()
for index, row in ahdb_dimers_filtered.iterrows():
    sequence = row["Sequence"]
    pep1, pep2 = sequence.split("GGSGGGS")
    if pep1 in unique_monomers or pep2 in unique_monomers:
        ahdb_dimers_filtered_unique.drop(index, inplace=True)
    else:
        unique_monomers.add(pep1)
        unique_monomers.add(pep2)

ahdb_dimers_filtered_unique.reset_index(drop=True, inplace=True)

"""## Butina Clustering"""

matrix = matlist.blosum62
centroids_ahdb = []

ahdb_dimers_uniquepeptides_eff = pd.read_excel("data/ahdb_dimers_uniquepeptides_eff.xlsx", index_col=0)
clusters_uniquepeptides = cluster_peptides(ahdb_dimers_uniquepeptides_eff.index.values, 15, psa) # Distance of 15 or lower considered neighbors, psa function defined above used to calculate distance

for i in clusters_uniquepeptides:
    cluster_efficacy = ahdb_dimers_uniquepeptides_eff.iloc[np.asarray(i), :]
    most_effective = cluster_efficacy[['Predicted Efficacy']].idxmax()
    most_effective_index = ahdb_dimers_uniquepeptides_eff.loc[most_effective,:].index[0]
    centroids_ahdb.append(most_effective_index)

ahdb_dimers_uniquepeptides_eff_centroids = ahdb_dimers_uniquepeptides_eff.loc[centroids_ahdb, :]