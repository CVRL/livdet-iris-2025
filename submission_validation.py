"""
This script validates LivDet submission.
"""
import subprocess
import os
import pandas as pd
import sys

if __name__ == '__main__':
    submission = sys.argv[1]
    if not os.path.exists(submission):
        print("INVALID SUBMISSION: Submission file does not exist")
        exit(1)

    subprocess.run(["python", submission, "sample.csv", "output.csv"], shell=True)

    # check if output file exists
    if not os.path.exists("output.csv"):
        print("INVALID SUBMISSION: Output csv file was not created")
        exit(1)

    # check if header is correct
    output_csv = pd.read_csv("output.csv")
    input_csv = pd.read_csv("sample.csv")
    if not all(output_csv.columns.values == ['index', 'filename', 'score']):
        print("INVALID SUBMISSION: Output csv file has incorrect header. Csv should contain index, filename and score "
              "columns")
        exit(1)

    # check number of lines in output file
    if not len(output_csv) == len(input_csv):
        print("INVALID SUBMISSION: Output csv should contain results for all images in input csv")
        exit(1)

    # check range of scores
    if not all(output_csv['score'].between(0, 1)):
        print("INVALID SUBMISSION: Output csv file has incorrect score range. Scores should be floating point numbers "
              "between 0 and 1.")
        exit(1)

    print("VALID SUBMISSION")
