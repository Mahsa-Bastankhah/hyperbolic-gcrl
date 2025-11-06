### Simplified Instructions when running on Della
```
module load anaconda3/2023.3
conda create -n hyp python=3.10
conda activate hyp
pip install torch hypll wandb graphviz ipython matplotlib
```
if you haven't used wandb, export WANDB_MODE=offline when running slurm scripts


### How to Run Hyperbolic Experiments
Install requirements:
```
pip install -r requirements.txt
```

Run train.py to train Contrastive RL:
```
kernprof -l train.py --config_file configs/config_hyp.yaml
```
or run `scripts/train_hyperbolic.slurm`.

You can visualize the results on the Poincare disc (if you use embedding_dim=2) using `notebooks/HyperbolicVisualizations.ipynb`

You can run discrete_maze experiments with scripts/discrete_maze.slurm

### Set experiments
Run set experiments with scripts/train_set.slurm. Code is in early-stages for these experiments. You can find the code for training experiments in train_tree_set.py. Network parameters are defined in nets. Environments are defined in environments.

### Notebooks
Notebooks SetViz.ipynb can be used for visualizing the set experiments

GeodesicInterpolation,ipynb, HyperbolicVisualization.ipynb, and TreevsEuclidean.ipynb for visualizing discrete experiments tree/maze experiments

### Pair encoder
This is a new version of the code that learns an embedding for a pair of initial and goal state s,g. A trajectory is sampled from teh replay buffer and phi(s,g) , phi(w_1,w_2) are used as positive example when s -> w_1 -> w_2 -> g for a trajectory.

This code only runs for the continuous maze enviornment, in particular I instantiated a 10x1 emptry room (no walls). 

In ordrer to run the experiment do:

```
python train.py --config_file configs/config_hyp.yaml
```

In order to see the norm of the embeddings for different pairs of states, run the following:

```
python -m notebooks.visualizePairEncoder
```
You should specify which epoch you want to look at in the code.