# AL-enhanced-Surrogate_MLMC-Adequacy_Assessment

This code accompanies the paper *MLMC-based Adequacy Assessment with Active Learning trained Surrogate Models* by
Ruiqi Zhang and Simon Tindemans, submitted at ISGT Europe 2025. The source code is built on the top of Paper: **Multilevel monte carlo with surrogate models for resource adequacy assessment**.

## Dependencies
modAL and sklearn packages are applied to train the random forest models with active learning strategy.
Two non-standard Python packages are required to run the code: `quadprog` and `gen_adequacy`.

## Quick start
1. `surrogate_model_data_set_generation.ipynb` generates test set for analyzing surrogate model accuracy and correlation.
2. `Results-for-ISGT2025-paper_part1_SurrogateModelTraining.ipynb` analyzes surrogate model performance training with active learning.
3. `Results-for-ISGT2025-paper_part2_MLMC_simulation.ipynb` performs the case study of MLMC-based adequacy assessment with active learning trained surrogate models

