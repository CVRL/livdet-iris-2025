# Iris Liveness Detection Competition (LivDet-Iris) 2025

Official repository for the 2025 edition of the Iris Liveness Detection Competition (LivDet-Iris)

### Table of contents
* [Summary](#summary)
* [Competition Tasks](#tasks)
* [Leaderboard](#leaderboard)
* [Datasets](#datasets)
* [Citations and Papers](#citations)
* [Acknowledgments](#acknowledgments)

<a name="summary"/></a>
### Summary

LivDet-Iris 2025 serves as the sixth edition of the iris liveness detection competition in the LivDet-Iris series. Held every two to three years, the competition aims to foster the development of robust algorithms capable of detecting a wide range of presentation attacks in iris biometrics. The 2025 edition obtained the largest number of submissions in the history of the competition: ten algorithms from five institutions, and one complete iris recognition system from one company. LivDet-Iris 2025 also introduced new tasks compared to previous editions: (Task 1) a benchmark offered by an industry partner, (Task 2) a new attack type, morphed iris images, in which two different-identity samples were blended into one image, and (Task 3) evaluation of presentation attack detection robustness against advanced manufacturing techniques for textured contact lenses. This edition for the first time in the series offers a systematic testing of a commercial iris recognition system (software and hardware) using physical artifacts presented to the sensor. **Dermalog-Iris** team submitted algorithms that won all tasks, achieving the area under the ROC curve of 90.57, 68.23 and 99.99 in tasks 1, 2, and 3, respectively. Additionally, we include results for baseline algorithms, based on modern deep convolutional neural networks and trained with all available public datasets of iris images representing bona fide samples and anomalies (physical attacks, eye diseases, post-mortem cases, and synthetically-generated iris images). Test samples created for tasks 2 and 3, along with the baseline models, are made available with this paper to offer the state-of-the-art benchmark for iris liveness detection assessment.

<a name="tasks"/></a>
### Competition Tasks

- **Part 1 (Algorithms)** involves the evaluation of the software solutions (submitted to the organizers) in three tasks, in which large datasets of iris images representing bona fide samples and various anomalies were used:
	- **Task 1: Industry Partner's Tests:** The industry partner, PayEye, Poland, evaluated all submissions using a sequestered dataset that reflects the most prevalent physical attacks observed in real-world iris recognition-based payment services. The presentation attack instruments (PAIs) included in this task are paper printouts, irises displayed on an e-book reader, artificial eyes, doll eyes, mannequin eyes, as well as samples synthesized using Generative Adversarial Networks (GANs);
	- **Task 2: Deep Learning-Aided Iris Morphing:** Submissions were tested against morphed iris samples, prepared by compositing two iris images representing two identities, with the seams caused by the compositing process "smoothed" by a diffusion model to increase the visual realism of such morphed samples;
	- **Task 3: Robustness to Advanced Textured Contact Lens (TCL) Manufacturing:** focused on assessing the robustness of liveness detection methods against modern manufacturing techniques used to produce TCL brands, including high-resolution printing, multi-layered designs, and improved pigmentation. This task allowed to assess how well current methods can detect these next-generation lenses and examine the community's readiness in addressing this evolving threat.
- **Part 2 (Systems)** involves the systematic testing of submitted iris recognition systems based on physical artifacts presented to the sensors by a laboratory staff.

<a name="leaderboard"/></a>
### Leaderboard

<a name="datasets"/></a>
### Datasets

<a name="citations"/></a>
### Citations and Papers

IJCB 2025 paper (if accepted): **[IEEEXplore](https://ieeexplore.ieee.org/) | [ArXiv](https://arxiv.org/)**.

Bibtex:

```

```

<a name="acknowledgments"/></a>
### Acknowledgments]




