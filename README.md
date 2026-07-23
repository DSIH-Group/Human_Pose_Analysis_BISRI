# Human Pose Analysis
 
## Overview
 
Human pose estimation in the OR has largely focused on detecting and localizing individual
clinicians within single frames, leaving OR team dynamics across the course of a procedure
comparatively unexplored. In this work, we investigate whether unsupervised clustering of pose
and spatial-configuration data can reveal underlying patterns in surgical team behavior, using
the MVOR dataset, the first publicly available multi-view RGB-D dataset captured during real
clinical interventions. Since MVOR provides no procedure-type or team-behavior annotations,
we manually organize its 1061 3D pose annotations into 24 usable operations and construct a
64-dimensional per-frame representation spanning three facets: individual pose (Geometric Pose
Descriptor), interperson interaction (pose diversity and pairwise clinician distance), and group-
level spatial configuration relative to the operating table. Operations, treated as multivariate
time series, are compared using normalized Dynamic Time Warping (nDTW) and clustered with
hierarchical clustering and k-means with DTW Barycenter Averaging (DBA); spectral cluster-
ing is evaluated but ultimately discarded after experiments indicate the data lacks non-convex
structure. Since no ground-truth procedure labels exist, we validate cluster assignments against
manual annotations consisting of C-arm presence and patient orientation, measured via the Ad-
justed Rand Index (ARI). While clustering on the full feature set yields only weak agreement
with annotations, ablation studies reveal that the interperson feature group alone drives the
strongest and most consistent signal, achieving an ARI of 0.55 against C-arm presence once
short, low-information operations are filtered out. We further show that the drivers of this
signal shift unexpectedly between pose-diversity and clinician distance subfeatures depending
on the operation-length filter applied. These findings suggest that interperson spatial dynamics,
rather than individual pose or global room configuration, carry the most discriminative infor-
mation about surgical activity, while also highlighting the data and annotation limitations that
constrain more definitive conclusions.

To read more about project consult `Human Pose Analysis Report.pdf`
 
## Pipeline
 
The notebooks/scripts below **must be run in this order** — each stage consumes
the outputs of the previous one.
 
### 1. `extract.ipynb` — Preprocessing
Parses the raw MVOR pose annotations (`camma_mvor_2018.json`) into per-frame and
per-operation structures.
- **Outputs:** `poses_per_operation.json`, `cleanedAnnots.pkl`
### 2. `gpd.ipynb` — Geometric Pose Descriptor (GPD)
Computes the GPD frame vectors (raw ~749 dims) and reduces dimensionality via
PCA (retaining 90% variance, ~50 dims).
- **Outputs:** `gpdAnnots.pkl` — GPD vector sequences per operation
### 3. `frameVector.ipynb` — Full Frame Vector Construction
Combines GPD features with the interperson (pose diversity + joint distance)
and group/table feature groups into a single per-frame vector.
- **Outputs:** `frameVectorAnnots.pkl` — full frame vector sequences per operation
### 4. `efa.ipynb` + `parralel_analysis.R` / `parralel_analysis_full.R` — Exploratory Factor Analysis
Prepares frame-level data to test whether the interperson feature group
contributes independent signal beyond GPD/table features. EFA and parallel
analysis are run in R.
- **Outputs:** `frameVectorDf.csv`, `completeFrameMatrix.csv` (inputs to the R scripts)
- `parralel_analysis.R`: parallel analysis + EFA restricted to multi-clinician frames
- `parralel_analysis_full.R`: parallel analysis + EFA on all frames
### 5. `normalize.ipynb` — Normality Checks and Normalization
Checks the normality of feature distributions and normalizes operation-level
sequences.
- **Outputs:**
  - `operationList.pkl` — normalized feature vector sequences (primary output)
  - `frameMatrixScaled.pkl`, `weightedFrames.pkl`, `ahpFrames.pkl` — weighting variants
  - `first90percent.pkl`, `last90percent.pkl`, `operation_chunks.pkl` — ablation inputs
### 6. `cluster.ipynb` — Clustering
Clusters operations using hierarchical clustering (HC) and k-means with DTW
Barycenter Averaging (DBA), using normalized DTW (nDTW) as the alignment/distance
metric. Validates against manual ground truth using ARI, with cross-method ARI as supporting diagnostics.
 
### 7. `ablation.ipynb` — Ablation Studies
- Filtering Short Operations
- Feature group subsetting
- Feature weighing approaches
  
## Repository Structure
 
```
.
├── Human Pose Analysis Report.pdf # Full report of project containing operation representation methodology, and our results, along with visualization of the clustering outcomes.
├── extract.ipynb
├── gpd.ipynb
├── frameVector.ipynb
├── efa.ipynb
├── parralel_analysis.R
├── parralel_analysis_full.R
├── normalize.ipynb
├── cluster.ipynb
├── ablation.ipynb
├── camma_mvor_2018.json     # raw MVOR keypoint annotations (input)
├── op_to_day_map.*          # maps operations to characteristics: (#frames, ablation indices, manual annotations)
├── utils/
│   ├── general_utils.py        # helpers used in extract.ipynb
│   ├── gpd_utils.py             # helpers for constructing the GPD frame vector
│   ├── poseDiversity_utils.py   # constructs the pose-diversity feature subgroup
│   ├── scene_graph_utils.py     # computes table features and joint-distance subgroup
│   ├── cluster_utils.py         # k-means with normalized DBA implementation
│   └── DBA / DBA_multivariate.py# original DBA, modified to add normalization
```

## Running the Pipeline
 
1. Run notebooks in the order listed above (1 → 7); each stage's `.pkl`/`.json`
   outputs are consumed by the next.
2. (optional) For the EFA stage, export `frameVectorDf.csv` and `completeFrameMatrix.csv`
   from `efa.ipynb`, then run the corresponding `.R` script.
4. Final clustering results and ablation outputs are produced by
   `cluster.ipynb` and `ablation.ipynb` respectively.
