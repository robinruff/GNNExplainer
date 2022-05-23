
*The code in this repository is now integrated into [KGCNN repository](https://github.com/aimat-lab/gcnn_keras/blob/master/kgcnn/literature/GNNExplain.py).*

# GNNExplainer

This repository provides a way to explain Keras/Tensorflow Graph Neural Network decisions.
The basic idea behind the generation method for explanations is heavily inspired by the paper ["GNNExplainer: Generating Explanations for Graph Neural Networks" by Ying et al.](https://arxiv.org/abs/1903.03894), although it differs in some aspects.

## Structure of this Repository

```
.
├── LICENSE                           # License file
├── README.md                         # This README
├── abschlussbericht.ipynb            # Final report on the project
├── examples                          
│   ├── cora                          
│   │   ├── data                      # Cora dataset
│   │   ├── dataloader.py             # Data loading utility
│   │   └── example_notebook.ipynb    # Example for CORA 
│   └── mutagenicity                  
│       ├── data                      # Mutagenicity dataset
│       ├── dataloader.py             # Data loading utility
│       ├── example_notebook.ipynb    # Mask on input
│       ├── example_notebook2.ipynb   # Mask on GNN operation
│       └── example_notebook3.ipynb   # Multi-Instance explanations 
├── gnnx                              # gnnx-Package
│   ├── __init__.py
│   └── gnnx.py                       # Core implemenation
├── pyproject.toml
└── setup.py
```


## Installation of the `gnnx` package

```
pip install -e .
```

## Example Workflow

```python
# Configuration
compile_options = {'loss': 'binary_crossentropy'}
fit_options = {'epochs': 100}
config = {
	'edge_mask_loss_weight': 0.01,    # λ_M
	'edge_mask_norm_ord': 1,          # p_M
	'feature_mask_loss_weight': 0.01, # λ_F
	'feature_mask_norm_ord': 1,       # p_F
	'node_mask_loss_weight': 0,       # λ_N
	'node_mask_norm_ord': 1}          # p_N
# Setup. Instantiating the GNNExplainer
explainer = gnnx.GNNExplainer(
    gnn, # Implements gnnx.GNNInterface
	compile_options=compile_options,
	fit_options=fit_options,
	gnnexplaineroptimizer_options=config)
# Explaining Decisions
explainer.explain(input_instance) 
explanation = explainer.get_explanation() 
explainer.present_explanation(explanation) 
```

For more examples look inside the [examples directory](https://github.com/robinruff/GNNExplainer/tree/main/examples).

## Class Diagram for `gnnx` package 

![Class diagram](https://github.com/robinruff/GNNExplainer/blob/main/class_diagram.svg)
