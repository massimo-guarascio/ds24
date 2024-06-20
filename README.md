# Beyond the Horizon: Using Mixture of Experts for Domain Agnostic Fake News Detection

This is the accompanying code for the paper "Beyond the Horizon: Using Mixture of Experts for Domain Agnostic Fake News Detection".

The repository contains all code to re-execute our models. You need Pytorch >= 1.5 to run it. 

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