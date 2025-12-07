# Mitigating Negative Feedback in Sequential Recommendations with Barlow Twins

This repository contains code for *Mitigating Negative Feedback in Sequential Recommendations with Barlow Twins* paper.

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```

For configuration we use [Hydra](https://hydra.cc/). Parameters are specified in [config files](runs/configs/).

Example of run for ML1M dataset:

```sh
cd runs
python run_negbt.py 
```
