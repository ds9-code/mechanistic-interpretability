# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pngs/brainiac.jpeg" width="800" alt="BrainIAC_V2 Logo"/>
</p>

## Overview

BrainIAC (Brain Imaging Adaptive Core) is vision based foundation model for generalized structural Brain MRI analysis. This repository provides the BrainIAC and downstream model checkpoints, with training/inference pipeline across all downstream tasks. Checkout the [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11643205/)


## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.0+ (for GPU training)
- python >= 3.9


### Setup Environment

```bash
# Clone the repository
git clone https://github.com/YourUsername/BrainIAC_V2.git
cd BrainIAC

# Create conda environment

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r requirements.txt
```

## Model Checkpoints

Download the BrainIAC weights and dowstream model checkpoints and place it in `./src/checkpoints/`:

**Download Link:** [Model Checkpoints](https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=r5nyejyo&dl=0)



## Quick Start

See [quickstart.ipynb](./src/quickstart.ipynb) to get started on how to preprocess data, load BrainIAC to extract features, generate and visualize saliency maps. We provide data samples from publicly available [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) [License](https://creativecommons.org/licenses/by/4.0/) (with no modifications to the provided preprocessed images) and the [Pixar](https://openneuro.org/datasets/ds000228/versions/1.1.1)  [License](https://creativecommons.org/public-domain/cc0/) dataset in the [sample_data](src/data/sample/processed/) directory. 


## Train and Infer Downstream Models

- [Brain Age Prediction](./docs/downstream_tasks/brain_age_prediction.md)
- [IDH Mutation Classification](./docs/downstream_tasks/idh_mutation_classification.md)
- [Mild Cognitive Impairment Classification](./docs/downstream_tasks/mild_cognitive_impairment_classification.md)
- [Diffuse Glioma Overall Survival Prediction](./docs/downstream_tasks/diffuse_glioma_overall_survival.md)
- [MR Sequence Classification](./docs/downstream_tasks/MR_sequence_classification.md)
- [Time to Stroke Prediction](./docs/downstream_tasks/timetostroke_prediction.md)
- [Tumor Segmentation](./docs/downstream_tasks/tumor_segmentation.md)


## Application Demos

Test BrainIAC's downstream models on your own data with our interactive demos, hosted on hugging face spaces!

- [**Brain Age Prediction**](https://huggingface.co/spaces/Divytak/BrainIAC-Brainage-V0): Brain age prediction from structural T1 MRI scans
  - Upload T1w MRI scan to get the brain age

- [**Mild Cognitive Impairment (MCI) Classification**](https://huggingface.co/spaces/Divytak/BrainIAC-MildCognitiveImpairment_Classification): Mild cognitive impairment risk prediction from structural T1 MRI scans
  - Upload T1w MRI scan to get the MCI risk score

interactive demos coming soon for other downstream tasks!

## Docker

For local deployment, run BrainIAC downstream model application using Docker:

***IDH app***
```bash
docker run -p 7860:7860 --name brainiac-idh-app divyanshutak1/brainiac-idh:latest
```

***MCI app***
```bash
docker run -p 7860:7860 divyanshutak1/brainiac-mci:latest
```



## Citation

```bibtex
@article{tak2024brainiac,
    title={BrainIAC: A Foundation Model for Generalized Brain MRI Analysis},
    author={Tak, Divyanshu and others},
    journal={medRxiv},
    year={2024},
    doi={10.1101/2024.12.02.24317992}
}
```

## License

BrainIAC is released under the CC BY-NC License. See [LICENSE](LICENSE) for details.


