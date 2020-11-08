# PatentMatch: A Dataset for Matching Patent Claims & Prior Art
This repository accompanies the paper "PatentMatch: A Dataset for Matching Patent Claims & Prior Art" by Julian Risch, Nicolas Alder, Christoph Hewel and Ralf Krestel. It shows how to use the dataset to train a BERT model on the dataset with the help of the [FARM framework](https://github.com/deepset-ai/FARM). The dataset can be downloaded [here](https://hpi.de/naumann/projects/web-science/deep-learning-for-text/patentmatch.html). The paper is currently under single-blind review for publication at [CHIIR 2021](https://acm-chiir.github.io/chiir2021/).

Install the framework by running the following commands:
```git clone https://github.com/deepset-ai/FARM.git
cd FARM
pip install -r requirements.txt
pip install --editable .
```
To run the example, copy the dataset into the ```data/``` folder and execute the command:

```python examples/text_pair_classification.py```
