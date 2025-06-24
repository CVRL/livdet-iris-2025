# Iris Liveness Detection Competition (LivDet-Iris) 2025

Official repository for the 2025 edition of the Iris Liveness Detection Competition (LivDet-Iris).

### Quick resources 
- Official LivDet-Iris 2025 webpage: [https://livdet-iris.org/2025](https://livdet-iris.org/2025)
- LivDet competition series: [https://livdet.org](https://livdet.org)
- IJCB 2025 paper (if accepted): [IEEEXplore](https://ieeexplore.ieee.org/) | [ArXiv](https://arxiv.org/).

### Table of contents
* [Summary](#summary)
* [Competition Tasks](#tasks)
* [Leaderboard](#leaderboard)
* [Datasets](#datasets)
* [Citations](#citations)
* [Acknowledgments](#acknowledgments)

<a name="summary"/></a>
### Summary

LivDet-Iris 2025 serves as the sixth edition of the iris liveness detection competition in the LivDet-Iris series. Held every two to three years, the competition aims to foster the development of robust algorithms capable of detecting a wide range of physically- and digitally-presented attacks in iris biometrics. The 2025 edition obtained the largest number of submissions in the history of the competition: ten algorithms from five institutions, and one commercial iris recognition system. 

<a name="tasks"/></a>
### Competition Tasks

- **Part 1 (Algorithms)** involves the evaluation of the software solutions (submitted to the organizers) in three tasks, in which large datasets of iris images representing bona fide samples and various anomalies were used:
  
	- **Task 1: Industry Partner's Tests:** The industry partner, PayEye, Poland, evaluated all submissions using a sequestered dataset that reflects the most prevalent physical attacks observed in real-world iris recognition-based payment services. The presentation attack instruments (PAIs) included in this task are paper printouts, irises displayed on an e-book reader, artificial eyes, doll eyes, mannequin eyes, as well as samples synthesized using Generative Adversarial Networks (GANs);
	- **Task 2: Deep Learning-Aided Iris Morphing:** Submissions were tested against morphed iris samples, prepared by compositing two iris images representing two identities, with the seams caused by the compositing process "smoothed" by a diffusion model to increase the visual realism of such morphed samples;
	- **Task 3: Robustness to Advanced Textured Contact Lens (TCL) Manufacturing:** focused on assessing the robustness of liveness detection methods against modern manufacturing techniques used to produce TCL brands, including high-resolution printing, multi-layered designs, and improved pigmentation. This task allowed to assess how well current methods can detect these next-generation lenses and examine the community's readiness in addressing this evolving threat.
- **Part 2 (Systems)** involves the systematic testing of submitted iris recognition systems based on physical artifacts presented to the sensors by a laboratory staff.

<a name="leaderboard"/></a>
### Leaderboard

(will be added after official announcement of the competition results)

<a name="datasets"/></a>
### Datasets

Instructions on how to obtain a copy of test data used in Task 2 can be found at the [Notre Dame's Computer Vision Research Lab webpage](https://cvrl.nd.edu/projects/data/) (search for ``LivDet-Iris 2025 Task 2`` dataset).

**Note:** the dataset will be added to the CVRL webpage after the LivDet-Iris paper is accepted.

To obtain a copy of the test data used in Task 3, you may contact livdet@gmail.com to request the release agreement.
The dataset consists of 564 bona fide iris images and 788 presentation attack samples collected from 7 subjects wearing textured contact lenses. These attack samples represent nine different contact lens brands and are categorized into two quality classes: High Quality and Pixelated. Specifically, 644 high-quality samples originate from six different contact lenses, while the remaining 144 pixelated samples are from three other contact lenses.

<a name="citations"/></a>
### Citation

```
@InProceedings{Mitcheff_IJCB_2025,
  author    = {
 Mitcheff, Mahsa and Hossain, Afzal and Webster, Samuel 
 and Karim, Siamul Khan and Roszczewska, Katarzyna and Tapia, Juan 
 and Stockhardt, Fabian and Gonzalez-Soler, Janier and Lim, Ji-Young 
 and Pollok, Mirko and Kreuzer, Felix and Wang, Caiyong and Li, Lin 
 and Guo, Fukang and Gu, Jiayin and Pal, Debasmita and Farmanifard, Parisa 
 and Sharma, Renu and Ross, Arun and Sharma, Geetanjali and Ashwani, Shubham 
 and Nigam, Aditya and Ramachandra, Raghavendra and Igene, Lambert 
 and Dykes, Jesse and Sawilska, Ada and  Dzieniszewska, Aleksandra 
 and Januszkiewicz, Jakub and Bartuzi-Trokielewicz, Ewelina 
 and Martinek, Alicja and Trokielewicz, Mateusz and Kordas, Adrian 
 and Bowyer, Kevin and Schuckers, Stephanie and Czajka, Adam},
  booktitle = {2025 IEEE International Joint Conference on Biometrics (IJCB), Osaka, Japan},
  title     = {{Iris Liveness Detection Competition (LivDet-Iris) –- The 2025 Edition}},
  year      = {2025},
  pages     = {1-10},
  doi       = {},
  keywords  = {},
}
```

<a name="acknowledgments"/></a>
### Acknowledgments

This material is based upon work supported by the U.S. National Science Foundation under grants No. 2237880 and 1650503. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. National Science Foundation. Caiyong Wang was funded by the Beijing Natural Science Foundation (4242018).




