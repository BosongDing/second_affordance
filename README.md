# Second Affordance
A repository for reproducing the results presented in ROMAN-2025 submission.

> **Abstract:** 

## Folder and file descriptions
Run the experiments in `run.sh` to reproduce the results. Use `--random_seed=` for different random seeds.\
resnetXX.py --> Stack\
resnetXX_3cam_3Net.py --> 3c 3n\
resnetXX_3cam_6Net.py --> 3c 6n\
resnetXX_centercam_1Net.py --> cc 1n\
resnetXX_centercam_2Net.py --> cc2n

the dataset can be downloaded at https://www.crossvalidate.me/datasets.html
and the files should have the following structure: 
```
workspace
├── code (this repo) 
├── dataset 
│ ├── 0_woodencube 
│ ├── 1_peartoy 
│ ├── ... 
└── processed_data 
```

the pre-trained weight of best performance model :ResNet50 CC-1N can be downloaded at https://drive.google.com/file/d/1ZZ4mJH9ekiQbJozsWCunxFR9W3uwS3mX/view?usp=sharing

|      | res18                   | res50                   | res101                  |
|------|-------------------------|-------------------------|-------------------------|
| Stack| 76.68 ± 1.31          | 80.62 ± 2.70          | 79.00 ± 1.79          |
| 3c 3n| 81.56 ± 2.247          | 86.09 ± 3.02          |80.87 ± 1.01          |
| 3c 6n| 82.86 ± 3.35         | 73.90 ± 1.30          | 71.28 ± 1.39          |
| cc 1n| 86.06 ± 2.05          | 90.78 ± 4.18         | 85.15 ± 5.24          |
| cc 2n| 78.06 ± 2.57          | 76.31 ± 5.10          | 72.43 ± 1.99          |
