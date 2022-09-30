# cdml

## Overview

solution submitted to the [Cross-Domain MetaDL challenge](https://metalearning.chalearn.org/) in [NeurIPS 2022 competition track](https://neurips.cc/Conferences/2022/CompetitionTrack).

## Description

We have based our solution to the freestyle league on the general framework of the metadelta++ [1], the winnig solution in NeurIPS 2021 MetaDL competition. Additionally, our final model consists of the following:

- **Ensemble backbones**: We fine-tune a pre-trained seresnet101aa (with anti aliasing) architecture using set 0 in the given meta-training time. We use snapshot ensembling to retain two versions of the seresnet101aa. Additionally, we also upload a resnet50 model trained offline on set 0.
- **Contrastive loss**: We apply triple-margin loss in addition to the cross-entropy based classification loss when fine-tuning the backbone. 
- **Self-Optimal-Transport**: In meta-classification stage, we apply the Self-Optimal-Transport (SOT) [3] feature transform to the ensemble sets of support and querry set features. SOT provides a mapping to the features that promotes downstream matching and grouping tasks. The implimentation is borrowed from [here](https://github.com/DanielShalam/SOT).

Our solution to the free-style league follows the protonet [2] framework, where we add contrastive loss in addition to the episodic loss when training the backbone.  



## Requirements

Please follow the [official website](https://github.com/DustinCarrion/cd-metadl/tree/8c6128120ab8aac331c958b2965d42747d9dbdeb) to set-up environments.  

## Run the code under competition setting

Please follow the [official website](https://github.com/DustinCarrion/cd-metadl/tree/8c6128120ab8aac331c958b2965d42747d9dbdeb) to run the codes.  
For example, cd to the folder of the cd-metadl, and run the following command:
```
cd path/to/cd-metadl
python -m cdmetadl.run --input_data_dir=public_data --submission_dir=path/to/this/folder --output_dir_ingestion=ingestion_output --output_dir_scoring=scoring_output --verbose=False --overwrite_previous_results=True --test_tasks_per_dataset=10
```

Remember to replace the `path/to/cd-metadl` and `path/to/this/folder` to your settings.

## Notes on repository branches:
The cdml (default) branch consist of the code to train a meta-learning model using an ensemble of publicly available online pre-trained models as well as a pre-trained model that was trained offline on Set 0 (named model3) in the code. Please reach out to us if you would like to have access to the trained model weights for model3. Alternatively, model3 can also be trained by from scratch by running the code in the branch: backbone_pretraining_cdml. 

The cdml_2 branch consist of the code to train the model from scratch without using any pre-trained model for the meta-learning league.

## references
- [1] : [Chen, Yudong, Chaoyu Guan, Zhikun Wei, Xin Wang, and Wenwu Zhu. "Metadelta: A meta-learning system for few-shot image classification." In AAAI Workshop on Meta-Learning and MetaDL Challenge, pp. 17-28. PMLR, 2021](https://arxiv.org/abs/2102.10744)
- [2] : [Snell, Jake, Kevin Swersky, and Richard Zemel. "Prototypical networks for few-shot learning." Advances in neural information processing systems 30 (2017)](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)
- [3] : [Shalam, Daniel, and Simon Korman. "The Self-Optimal-Transport Feature Transform." arXiv preprint arXiv:2204.03065 (2022)](https://arxiv.org/abs/2204.03065)
