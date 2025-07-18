import csv
from sklearn.metrics import roc_auc_score

def evaluate_model(csv_path):
    # Load in CSV data
    truths = []
    preds = []
    num_nonresponses = 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader: # truth, pred, type
            truths.append(float(row[0]))
            pred = float(row[1])
            if pred == -1.0: # For simplicity, we marked samples with non-responses (eg. failure to predict, error, cause crash) as a predicted score of -1.0. Note that all Part 1 models had a 100% response rate and therefore was not reported in the paper.
                num_nonresponses += 1
                pred = 1.0 # For computing metrics, we consider non-responses as a prediction of attack (ie worst case, or otherwise a failure to accept bona fide)
            preds.append(pred)

    # Compute AUROC
    auroc = roc_auc_score(truths, preds)

    # Compute decisions at Acceptance Rate = 0.5
    threshold = 0.5
    decisions = [1.0 if p >= threshold else 0.0 for p in preds]

    # Compute BPCER - rate of bona fides falsely predicted as attacks
    num_bonafides = sum(1 for truth in truths if truth == 0.0)
    false_positives = sum((truth == 0.0 and dec == 1.0) for truth, dec in zip(truths, decisions))
    bpcer = false_positives / num_bonafides

    # Compute APCER - rate of attacks falsely predicted as bona fides
    num_attacks = sum(1 for truth in truths if truth == 1.0)
    false_negatives = sum((truth == 1.0 and dec == 0.0) for truth, dec in zip(truths, decisions))
    apcer = false_negatives / num_attacks

    # Compute NRR - rate of non-responses
    nrr = num_nonresponses / len(truths)

    return auroc, bpcer, apcer, nrr

# Example usage:
# auroc, bpcer, apcer, nrr = evaluate_model("path/to/your/file.csv")
# print(f"Results: ")
# print(f"\tAUROC: {auroc}")
# print(f"\tBPCER: {bpcer}")
# print(f"\tAPCER: {apcer}")
# print(f"\tNRR: {nrr}\n")