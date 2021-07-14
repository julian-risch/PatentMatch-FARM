# PatentMatch: A Dataset for Matching Patent Claims & Prior Art
This repository accompanies the paper "PatentMatch: A Dataset for Matching Patent Claims & Prior Art" by Julian Risch, Nicolas Alder, Christoph Hewel and Ralf Krestel. It shows how to use the dataset to train a BERT model on the dataset with the help of the [FARM framework](https://github.com/deepset-ai/FARM). The dataset can be downloaded [here](https://hpi.de/naumann/projects/web-science/paar-patent-analysis-and-retrieval/patentmatch.html). The paper is published on [arxiv.org](https://arxiv.org/abs/2012.13919) and in the proceedings of the PatentSemTech Workshop co-located with SIGIR 2021.

Install the framework by running the following commands:
```git clone https://github.com/deepset-ai/FARM.git
cd FARM
pip install -r requirements.txt
pip install --editable .
```
To run the example, copy the dataset into the ```data/``` folder and execute the command:

```python examples/text_pair_classification.py```

# Citation
If you use our work, please cite our paper [**PatentMatch: A Dataset for Matching Patent Claims & Prior Art**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020patentmatch.pdf) as follows:

    @article{risch2020match,
    title={{PatentMatch}: A Dataset for Matching Patent Claims with Prior Art},
    author={Risch, Julian and Alder, Nicolas and Hewel, Christoph and Krestel, Ralf},
    year={2020},
    journal = {ArXiv e-prints},
    eprint={2012.13919},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    }
