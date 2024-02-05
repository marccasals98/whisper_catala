# PYTHON VIRTUAL ENVIRONMENT

#!/bin/bash

venv_name="whisper_venv"

# Check if the virtual environment exists
if [ -d "$venv_name" ]; then
    echo "Virtual environment '$venv_name' already exists."
else
    # Create the virtual environment
    python3 -m venv "$venv_name"

    if [ $? -eq 0 ]; then
        echo "Virtual environment '$venv_name' created successfully."
    else
        echo "Error creating virtual environment '$venv_name'."
    fi
fi
source whisper_venv/bin/activate

pip install -r requirements.txt
# pip install accelerate -U
# pip install transformers[torch]
