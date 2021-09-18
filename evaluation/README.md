This folder consists evaluation of three approaches on clone detection task.
```
cd simian
docker build -t simian .

# Compute similarity for codeforces
docker run -v ~/contrastive-learning-framework/data/codeforces:/data -i -t simian
# Compute similarity for poj_104
docker run -v ~/contrastive-learning-framework/data/poj_104:/data -i -t simian
``` 