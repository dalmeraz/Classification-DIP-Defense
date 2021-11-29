# Classification-DIP-Defense
## Getting Started
First create new conda environment by running:
conda env create -f environment.yml

Then, install extra dependencies:
* pip install torchattacks
* pip install opencv-python
* pip install transformers
* conda install -c conda-forge tqdm

## Sources
For DIP library, we use a cloned version of https://github.com/DmitryUlyanov/deep-image-prior
For adverserial attacks, we use https://github.com/Harry24k/adversarial-attacks-pytorch which was installed through PyPi.
