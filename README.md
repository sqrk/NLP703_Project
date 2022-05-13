# NLP 703 Project - Dysarthric Speech Recognition: From Impaired to Understandable

To train the Conformer CTC and Quartznet, you need [nemo](https://github.com/NVIDIA/NeMo). It's recommended to use a fresh Conda env.

```
conda create --name nemo python==3.8
conda activate nemo

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```

You first need to build the manifests of each toolkit by running the python files in the corresponding directories (see the Manifests folder).

Checkpoints files were too large to be uploaded to the repo.
