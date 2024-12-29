# XMODE

This repository contains the code and resources for the paper [XMODE: Explainable Model for Data Extraction](https://arxiv.org/abs/2412.18428).

## Overview

XMODE is a framework designed to enhance the explainability of data extraction models. It provides tools and methodologies to interpret and visualize the decision-making process of machine learning models used in data extraction tasks.

## Repository Structure
> TODO Update the structure if needed
- `ArtWork/`: Contains artwork-related files and experiments.
- `dataset/`: Includes EHRXQA datasets used for evaluation.
- `experiments/`: Contains experimental scripts and results.
- `files/`: Stores images files of the EHRXQA datasets
- `preprocess/`: Scripts for preprocessing data.
- `reports/`: Stores reports files of the EHRXQA datasets
- `src/`: Source code for the XMODE framework.
- `tools/`: Utility scripts and tools.
- `README.md`: This file.
- `requirements.txt`: List of dependencies required to run the project.

## Getting Started

### Prerequisites

#### Ensure you have Python 3.8 or higher installed. Install the  required dependencies using:

```sh
pip install -r requirements.txt
```

#### Adding your own API Keys

Add the API keys for the following services in the `.env` file:

```sh
OPENAI_API_KEY={your_openai_api_key}
LANGCHAIN_API_KEY={your_langchain_api_key}
```

### Running the experiments:

#### Preparing the datasets:

1. Preparing the databases:
We have preprocessed the databases for both `Artwork` and `EHRXQA` datasets. You can download the preprocessed databases from the following links :

   - [art.db](https://drive.google.com/uc?export=download&id=1OMyab3ZbY92gKQ9FfC0z2c6SImakESrf) to the `ArtWork` directory.
   - [mimic_iv_cxr.db](https://drive.google.com/uc?export=download&id=19o7R_nZ3vSkVn8QXXoMVfe1zw6xqXv-N) to the root directory.

2. Preparing the images for the __EHRXQA__ dataset:

   - You can get the images from the [repository of EHRXQA](https://github.com/baeseongsu/ehrxqa) and copy them to the `files/` directory.

3. Preparing the images from the __Artwork__ dataset:

   - You can follow the instructions in [repository of CAESURA](https://github.com/DataManagementLab/caesura) to download the 100 first data and copy the `images` to the `ArtWork/data/` directory. 

You can get the images from the [repository of Artwork]() and copy them to the `files/` directory.

#### Run XMODE on the datasets:

1. On Artwork Dataset

```sh
python ArtWork/main.py
```

2. On EHRXQA Dataset

For getting the M3AE model working, you need to clone the [repository of M3AE](https://github.com/zhjohnchan/M3AE) and get the fine-tuned model from this [this thread](https://github.com/baeseongsu/ehrxqa/issues/7#issuecomment-2245718989). After the M3AE model service is deployed, you can run the following command:

```sh
python main_m3ae_EHRXQA.py
```

### Contributing

We welcome contributions to improve XMODE. Please fork the repository and submit pull requests.

### License
> TODO Add the license

### Citation

If you use this code in your research, please cite the paper:

```
@misc{nooralahzadeh2024explainablemultimodaldataexploration,
      title={Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent}, 
      author={Farhad Nooralahzadeh and Yi Zhang and Jonathan Furst and Kurt Stockinger},
      year={2024},
      eprint={2412.18428},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.18428}, 
}
```