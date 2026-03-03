MPL: Match-guided Prototype Learning for Few-shot Action Recognition

<img width="1269" height="449" alt="image" src="https://github.com/user-attachments/assets/3aca82aa-8a7c-4a42-8095-2030b52fd3a9" />

## Installation

Requirements:
- Python>=3.6
- torch>=1.5
- torchvision (version corresponding with torch)
- simplejson==3.11.1
- decord>=0.6.0
- pyyaml
- einops
- oss2
- psutil
- tqdm
- pandas

Or you can create environments with the following command:
```
conda env create -f environment.yaml
```

## Running
The entry file for all the runs are `runs/run.py`. 

The codebase can be run by:
```
python runs/run.py --cfg configs/projects/MPL/kinetics100/MPL_K100_1shot_v1.yaml
```
