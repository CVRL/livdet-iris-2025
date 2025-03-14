# LivDet-Iris 2025 Submission Package


### Submission script ###

The submission should be limited to one Python script called `submission_xyz.py` where
`xyz` is the participant identifier. It should accept two command line input arguments: path to
input csv and path to output csv.

Example command execution:

```
python submission.py input.csv output.csv
```

### Data Format ###

The images used in tests are in BMP format. Each image is an ISO complaint iris image (8 bit
grayscale, image resolution 640x480 pixels). After registration the participants will obtain an example image representing each task in the competition to make sure that the image format is known.

### Files ###

Input CSV file contains columns with index and paths to an ISO compliant image (see `sample.csv` for an example CSV). Output CSV file should contain columns with index, file path and the liveness score.

### Liveness Score Generation ###

The score should represent the probability (floating point number) between 0.0 and 1.0 that the
image is a presentation attack, where 0.0 means certainly not an attack, and 1.0 means certainly
an attack. If the image cannot be processed, return `None`.

### Validation ###

Before submitting your solution validate your submission with `submission_validation.py`. Validation script usage:

```
python submission_validation.py submission_xyz.py
```