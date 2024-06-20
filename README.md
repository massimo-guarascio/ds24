# MoDA-FND : A Domain Agnostic Deep Fake News Detector based on Mixture of Experts

This is the accompanying code for the paper "Beyond the Horizon: Using Mixture of Experts for Domain Agnostic Fake News Detection".

The repository contains all code to re-execute our models. You need Pytorch >= 1.5 to run it. 

## Description

MoDA-FND is framework that aims to integrate various domains to offer enhanced predictions for new ones. Specifically, the approach involves learning a distinct model for each domain and refining them through domain-specific adaptation procedures. The predictions of these refined models are hence blended using a Mixture of Experts approach, which allows for selecting the most reliable for predicting the new examples. The proposed approach is fully cross-domain and does not necessitate retraining or fine-tuning when encountering new domains, thus streamlining the adaptation process and ensuring scalability across diverse data landscapes.

## Run the code

Just execute

```
# train base models
python train.py -c config_crossdomain_moe_base_models.json
# update exp folder for base models in config file
python train.py -c config_crossdomain_moe.json
```



## Dataset

Preprocess each dataset saving the dataframe in parquet format and with the following columns:

```
'text', 'text_emb', 'label'
```

- *text* is the news content
- *text_emb* is the precomputed embedding
- *label* is 1 for fake and 0 for real data 