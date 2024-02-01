# Catalan Whisper (Cluster Version)
Finetuning whisper with Catalan datasets to change the domain of the original model.

## What's new?
This is the BSC's Cluster version. For this reason, there will be some changes. 

1. **Dataset creation:**The dataset will be imported from the cluster instead of being downloaded from the HF Hub. 
2. **Model loading:** The model will also be loaded locally.



## Initialization

1. Creation and activation of the Python venv:
```bash
python3 -m venv /path/to/new/virtual/environment
```

```bash
source <venv>/bin/activate
```
2. Install the requirements. To do so, just install the files in ```requirements.txt```

```bash
pip install -r requirements.txt
```
3. Run the ```main.py``` either locally or in any cluster using SLURM (Or whatever you have there).


**Warning!**

The following libraries need to be installed manually from terminal:
```bash
pip install accelerate -U
```

```bash
pip install transformers[torch]
```

TODO: Solve this.

## Datasets
For the moment, we encompass the following Datasets:

### Common Voice
Common voice it's a dataset developed by Mozilla of people recording their voice through microphones. 

Catalan is the most recorded voice, summing up to 3.500h.
