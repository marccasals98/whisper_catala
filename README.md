# Catalan Whisper 
Finetuning whisper with Catalan datasets to change the domain of the original model.

## Colab's version
Initially, the code was designed to fine-tune whisper with the Hindi language. The code was run successfully in a Colab. However, Catalan's dataset in Common-Voice is one of the biggest datasets that exists, so, downloading directly in the Colab, fills up the available SSD in the Colab. 

To overcome this problem, what it needs to be done is, instead of downloading the Dataset, use streaming. This just downloads the parts of the dataset that we are going to use. 

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
