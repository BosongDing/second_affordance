# HUMANOIDS-2024
A repository for reproducing the results presented in HUMANOIDS-2024 submission.

> **Abstract:** 

## Folder and file descriptions
Run the experiments in `run.sh` to reproduce the results. Use `--random_seed=` for different random seeds.\
resnetXX.py --> Stack\
resnetXX_3cam_3Net.py --> 3c 3n\
resnetXX_3cam_6Net.py --> 3c 6n\
resnetXX_centercam_1Net.py --> cc 1n\
resnetXX_centercam_2Net.py --> cc2n

the dataset can be downloaded at https://www.crossvalidate.me/datasets.html
the file should have the following structure: 
```
workspace
├── code (this repo) 
├── dataset 
│ ├── 0_woodencube 
│ ├── 1_peartoy 
│ ├── ... 
└── processed_data 
```

the pre-trained weight of best performance model :ResNet50 1C-1N can be downloaded at https://drive.google.com/file/d/1ZZ4mJH9ekiQbJozsWCunxFR9W3uwS3mX/view?usp=sharing

|      | res18                   | res50                   | res101                  |
|------|-------------------------|-------------------------|-------------------------|
| Stack| 77.031 ± 1.344          | 81.125 ± 1.898          | 79.812 ± 1.514          |
| 3c 3n| 85.397 ± 1.156          | 80.008 ± 5.628          | 74.343 ± 2.892          |
| 3c 6n| 77.736 ± 2.247          | 69.113 ± 3.109          | 70.917 ± 2.648          |
| cc 1n| 82.460 ± 2.041          | 88.192 ± 2.570          | 80.400 ± 1.374          |
| cc 2n| 71.487 ± 2.362          | 70.721 ± 0.957          | 69.729 ± 0.526          |
