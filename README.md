# HOTSNet

This git repo contains work related to my capstone project for Udacity's Machine Learning Nanodegree.  The paper for this project can be found [here](./paper/HOTSNetPaper.pdf)

## Building the project

You are required to have a working Jupyter Notebook installation with Python 3.6.  A GPU is highly recommended to train the model.

The required dependecies are:

* numpy
* pandas
* matplotlib
* tqdm
* sklean
* tensorflow
* keras
* requests

### Obtaining the dataset

The dataset can be automatically downloaded and preprocessed by running rewrite_hots_data script while inside the `src` directory.

```
cd src
python rewrite_hots_data script.py
```

The preprocessing step will probably take about two hours.

### Building the model

The final model was built inside Jupyer using the [HotSPredictor](./src/CompositionCompare.ipynb) notebook.  The output from the session used in the paper has been exported to HTML and is preserved in the `notebook_export` directory along with exports from the other included notebooks.  

Two scripts were also used to optimize the model.  [fuzz_nets.py](./src/fuzz_nets.py)  will generate random networks to train indefinitely, [best_nets.py](../master/src/best_nets.py) is mostly the same code, but instead used a fixed dict of model parameters were discovered to work pretty well.

### Other notebooks

[CompositionCompare](./src/CompositionCompare.ipynb) - this notebook was used to create one of the more complicated visualations in the paper.  It uses the raw data set and is slow to count all the games; it will take 2 hours to run.

[HeroPCA](./src/HeroPCA.ipynb) - As part of my experiements, I performed PCA on the heroes.  Ultimately, it did not turn out to be useful enough to use, and so the paper does not go into too much depth about how it was performed.  It was still part of the process of developing the final model, and this notebook contains some of the code I used.
