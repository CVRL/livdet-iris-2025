# LivDet-Iris 2025 Evaluation Codes and Raw Scores 

## Part 1

The `part1/` directory contains evaluation code and raw scores for Part 1, Tasks 1-3. 

`part1/task1/`, `part1/task2/`, `part1/task3/` are directories containing raw model outputs used to compute the performance metrics of all competitor and baseline models. All raw outputs are formatted in `.csv` files with the following three columns: true class label (0 for bona fide, 1 for spoof), model output prediction (float on range 0.0 to 1.0), description of sample (e.g. attack, morph, pixelated, real/live).

`eval.py` contains code used to compute performance metrics for each model. The function contained, given a path to a raw output `.csv`, computes AUROC, BPCER @ 0.5 threshold, APCER @ 0.5 threshold, and a Non-Response Rate. 