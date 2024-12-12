# NeBC
This is the open-source codebase the paper: __Novelty Encouraged Beam Clustering Search for Multi-Objective De Novo Diverse Drug Design__. Please do not hesitate to contact us if you have any problems.

## Description
- Beam clustering search algorithm is proposed to generate molecules from scratch, balancing the diversity and the optimality of generated molecules simultaneously. A clustering strategy based on molecule structure similarities is utilized to partition the intermediate molecules into various groups for candidates selection, which ensures a balanced representation across different generation directions and proactively enhances the exploration of diverse molecular structures.
- An intrinsic reward is proposed to encourage the generation of novel molecules explicitly. Multiple expert models are constructed to predict molecule properties based on existing drug molecules. The variance of predictions obtained from these expert models is utilized to quantify the dissimilarity between the generated molecules and the existing training molecules, serving as a novelty reward.  
- The experimental results demonstrate the effectiveness of NeBC search, which has exhibited significant improvements not only in terms of optimality but also in the diversity and novelty of the generated molecules, surpassing the state-of-the-art algorithm QADD.

## Dependencies
- python==3.7.15
- torch==1.9.5
- numpy==1.21.5
- pickle==0.0.12
- rdkit==2022.9.3

## Pipeline
Five property predictions are trained by running the following code for five times independently.
```
python predict.py
```

The heuristic function is trained with
```
python heuristic.py
```

Molecules are generated with clustering beamsearch guided by the learned heuristic with 
```
python beamsearch_cluster.py
```

## Lisence
- All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

- The license gives permission for academic use only.
