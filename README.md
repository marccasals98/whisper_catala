# Catalan Whisper 
Finetuning whisper with Catalan datasets to change the domain of the original model.

test

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

3. Create environmental variable with the hugging face token: `export HF_TOKEN=hf_yOurToKEn'. You can check that it was properly created by writing:

```bash
printenv
```
 
4. Run the ```main.py``` either locally or in any cluster using SLURM (Or whatever you have there).


**Warning!**

The following libraries need to be installed manually from terminal:
```bash
pip install accelerate -U
```

```bash
pip install transformers[torch]
```

TODO: Solve this.

In theory now it is solved. 

## Datasets
For the moment, we encompass the following Datasets:

### Common Voice
Common voice it's a dataset developed by Mozilla of people recording their voice through microphones. 

Catalan is the most recorded voice, summing up to 3.500h.
