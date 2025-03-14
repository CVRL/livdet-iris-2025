"""
Example LivDet 2025 submission file.
The submission should accept two input arguments: path to input csv and path to output csv.

Input csv file contains columns with index and paths to an ISO compliant image.
Output csv file should contain columns with index, file path and the probability score.

Your model should predict the probability (floating point) between 0.0 and 1.0 that the image is a presentation attack.
0.0 means certainly not an attack, 1.0 means certainly an attack.
If the image cannot be processed, return None.

example run command: python submission.py input.csv output.csv
"""

import sys
import csv
import random

if __name__ == '__main__':
    csv_input = sys.argv[1]
    csv_output = sys.argv[2]
    with open(csv_input, newline='') as csv_input_file:
        with open(csv_output, 'w', newline='') as csv_output_file:
            reader = csv.reader(csv_input_file, delimiter=',')
            next(reader, None)  # skip the header
            writer = csv.writer(csv_output_file, delimiter=',')
            writer.writerow(['index', 'filename', 'score'])  # write header
            for row in reader:
                # load image
                # do something with the image
                score = random.uniform(0, 1)  # calculate probability score
                new_row = [*row, score]  # write result to output csv
                writer.writerow(new_row)
