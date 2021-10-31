## Instalation 

This code is written and tested with Matlab2020b and Python 3.8. 

### Requirements
- Python 3.8
- Matlab 2020b or Matlab 2021a (older versions might also work)
- pytorch >= 1.7.1
- torchvision 

### Getting started

```
# Create a conda environment for LOD
conda create -n LOD python=3.8
conda activate LOD

# Install Pytorch
# could use newer version of pytorch here
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

# Clone LOD
git clone https://github.com/huyvvo/LOD.git
cd LOD
mkdir data

# Install requirements
pip install -r requirements.txt

```
