# LivDet-Iris 2025 Submission Package


### Submission script ###

Each submission should be limited to one Python script called `submission_xyz_nnn.py` where
`xyz` is the participant identifier, and `nnn` is the algorithm identifier (if the participant submits more than one algorithm). The script should accept two command line input arguments: a path to the input CSV file, and a path to the output CSV file.

Example command line execution:

```
python submission_ND_001.py input.csv output.csv
```

### Data Format ###

All test images will be coded as lossless, 8 bit 640x480 pixel grayscale PNG, i.e., every pixel is represented by a single sample that is a grayscale level, where 0 is black and 255 is white. Further information on PNG format can be found in RFC2083. Images of authentic irises will conform to the `IMAGE_TYPE_VGA` image type, as defined in ISO/IEC 19794-6:2011, yet analogous conformance for spoof iris images is not guaranteed.

After registration the participants will obtain an example image representing each task in the competition to make sure that the image format is known.

### Files ###

Input CSV file contains columns with index and paths to an ISO compliant image (see `sample.csv` for an example CSV). Output CSV file should contain columns with index, file path and the liveness score.

### Presentation Attack Score Generation ###

The score should represent the probability (floating point number) between 0.0 and 1.0 that the
image is a presentation attack, where 0.0 means certainly not an attack, and 1.0 means certainly
an attack. If the image cannot be processed, return `None`.
                                
### Validation ###

Before submitting your algorithm validate your submission with `submission_validation.py` script.

Example command line execution:

```
python submission_validation.py submission_ND_001.py
```

### Test Machines Specifications ###

Machine used in **Task 1** evaluations: ...

Machine used in **Task 2** evaluations:


> CPU: AMD Ryzen Threadripper 3960X - 24-Core - 3.8 GHz / 4.5GHz Boost <br>
> RAM: 256GB DDR4 3200MHz (32GB x 8) <br>
> GPU: NVIDIA RTX 3090 24GB <br>

Machine used in **Task 2** evaluations: ...
