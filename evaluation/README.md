This folder consists evaluation of three approaches on clone detection task.

- To create `.yaml` files for `simian` evaluation run the following commands:
```
cd simian
docker build -t simian .

# Compute similarity for codeforces
docker run -v path-to-contrastive-learning-framework/data/codeforces:/data -i -t simian
# Compute similarity for poj_104
docker run -v path-to-contrastive-learning-framework/data/poj_104:/data -i -t simian
``` 

- To create embeddings using `infercode` evaluation run the following commands:
```
cd infercode
docker build -t infercode .

# Compute similarity for codeforces
docker run -v path-to-contrastive-learning-framework/data/codeforces:/data --gpus=all -i -t infercode
# Compute similarity for poj_104
docker run -v path-to-contrastive-learning-framework/data/poj_104:/data --gpus=all -i -t infercode
``` 

- To create embeddings using `transcoder` evaluation run the following commands:
```
cd trans_coder
docker build -t trans_coder .

# Compute similarity for codeforces
docker run -v path-to-contrastive-learning-framework/data/codeforces:/data_ --gpus=all -i -t trans_coder
# Compute similarity for poj_104
docker run -v path-to-contrastive-learning-framework/data/poj_104:/data_ --gpus=all -i -t simian
``` 
When all embeddings are computed, run script `evaluate.py` 