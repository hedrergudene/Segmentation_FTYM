# Semantic Segmentation w/ synthetic data

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Repo structure](#repo-structure)
- [3. Implementation details](#implementation-details)
- [4. Monitoring integration](#monitoring-integration)
- [5. Quickstart code](#quickstart-code)
- [6. License](#license)
---

## Introduction
Image segmentation is one of the most popular computer vision tasks, where every pixel is tagged as a specific label; it is the equivalent formulation of Named Entity Recognition (NER) tasks in natural language procesing:

![segm_ner](./images/segm_ner.jpg)

Some of its applications involve handling sensitive data, such as face parsing. In that direction, and in order to handle AI bias, the dataset [Fake It Till You Make It](https://microsoft.github.io/FaceSynthetics/) was designed. As per authors words,

> We demonstrate that it is possible to perform face-related computer vision in the wild using synthetic data alone. The community has long enjoyed the benefits of synthesizing training data with graphics, but the domain gap between real and synthetic data has remained a problem, especially for human faces. Researchers have tried to bridge this gap with data mixing, domain adaptation, and domain-adversarial training, but we show that it is possible to synthesize data with minimal domain gap, so that models trained on synthetic data generalize to real in-the-wild datasets.

Even though facial landmark detection is not covered in this use case, this dataset contains enough information to tackle that problem; you can find more info about modelling in this recent [survey paper](https://arxiv.org/abs/2101.10808).

## Repo structure

Since all scripts do have the same requirements and source folder, a unified structure for all of them is used, as illustrated below:


<details>
<summary>
Click here to find out!
</summary>

    ├── config                                 # Configuration files
    │   ├── experiment_config.yaml             # Configuration file for training and monitoring
    │   ├── single_cpu_config.yaml             # Configuration file for single-cpu thread
    │   ├── multi_cpu_config.yaml              # Configuration file for multi-cpu thread
    │   ├── single_gpu_config.yaml             # Configuration file for single-gpu thread
    │   └── multi_gpu_config.yaml              # Configuration file for multi-gpu thread
    |
    ├── input                                  # Dataset (generated during running)
    │   ├──train                               # Train split
    │   │  ├──images                           # Rendered image of a face
    │   │  │  ├──{frame_id_1}.png        
    │   │  │  ├──...
    │   │  │  └──{frame_id_n}.png        
    │   │  └──annotations                      # Segmentation image, where each pixel has an integer value
    │   │     ├──{frame_id_1}_seg.png        
    │   │     ├──...
    │   │     └──{frame_id_n}_seg.png        
    │   └──val                                 # Train split
    │      ├──images                           # Rendered image of a face
    │      │  ├──{frame_id_1}.png        
    │      │  ├──...
    │      │  └──{frame_id_m}.png        
    │      └──annotations                      # Segmentation image, where each pixel has an integer value
    │         ├──{frame_id_1}_seg.png        
    │         ├──...
    │         └──{frame_id_m}_seg.png     
    |
    ├── src                                    # Main methods to build scripts code
    │   ├── callbacks.py                       # Contains W&B logging
    │   ├── dataset.py                         # Method that structures and transforms data
    │   ├── fitter.py                          # Training, validation and storing loop wrapper
    │   ├── loss.py                            # Custom function to meet our needs during training
    │   ├── model.py                           # Core script containing the architecture of the model
    │   ├── setup.py                           # Helper methods to shorten main script length and make it more readable
    │   └── utils.py                           # Helper methods to control reproducibility
    │
    ├── requirements.txt                       # Libraries to be used and their versions
    └── train.py                               # Script to run model training
</details>



## Implementation details

* Data augmentation pipelines are based on [albumentations](https://albmentations.ai/) library.
* The chosen model architecture for this project is `nvidia/mit-b0` [SegFormer pretrained backbone :hugs:](https://huggingface.co/docs/transformers/model_doc/segformer) from [NVIDIA Research project ](https://github.com/NVlabs/SegFormer).
* Loss function is a combination of categorical crossentropy and a generalisation of the widely known Jaccard loss, based on [this paper](https://www.scitepress.org/Papers/2021/103040/103040.pdf). Data distribution has also be taken into consideration, by quantifying the percentage of pixels belonging to each label:

    ```python
    {0: 62.478843, 1: 8.605424, 2: 0.75495, 3: 0.052647, 4: 0.05276, 5: 0.109988, 6: 0.110996, 7: 0.189292, 8: 0.193591, 9: 0.108976, 10: 0.183389, 11: 0.209963, 12: 1.66163, 13:8.457067, 14: 0.67025, 15: 14.718491, 16: 0.164434, 17: 1.277309, 18: 0.0}
    ```
    
    As a result, loss function is weighted with inverse frequency ratios:
    
    * Weight for index 0: 0.0001658747314334
    * Weight for index 1: 0.0012043172954295
    * Weight for index 2: 0.0137276061200101
    * Weight for index 3: 0.1968507907700739
    * Weight for index 4: 0.1964319631162028
    * Weight for index 5: 0.0942255875320043
    * Weight for index 6: 0.0933697862761819
    * Weight for index 7: 0.0547495411674153
    * Weight for index 8: 0.0535336807499800
    * Weight for index 9: 0.0951002747093535
    * Weight for index 10: 0.0565119300908439
    * Weight for index 11: 0.0493595760815996
    * Weight for index 12: 0.0062370452717332
    * Weight for index 13: 0.0012254438559717
    * Weight for index 14: 0.0154623923498118
    * Weight for index 15: 0.0007041252515757
    * Weight for index 16: 0.0630263970748995
    * Weight for index 17: 0.0081136675554797
    * Weight for index 18: 0.0000000000000000


* Training loop has been built upon [accelerate :hugs:](https://github.com/huggingface/accelerate) library to be able to adapt our environment to a variety of computing resources.


## Monitoring integration
This experiment has been integrated with Weights and Biases to track all metrics, hyperparameters, callbacks and GPU performance. You only need to adapt the parameters in the `experiment_config.yaml` configuration file to keep track of the model training and evaluation. A model checkpoint trained in the full version of the dataset, together with training configuration, virtual machine specifications and model hyperparameters is provided [here](https://wandb.ai/azm630/segformer_FTYM?workspace=user-azm630). 

## Quickstart code
In the config section, you will find template configuration files to get up to speed with `accelerate`. As an example, the following terminal command will run the training script:

```bash
accelerate launch --config_file ./config/single_gpu_config.yaml train.py
```


## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).