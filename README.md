# Distributional-Dynamic-Movement-Primitives-D-DMPs

This repository contains the code for implementing Distributional DMPs (D-DMPs) in `pytorch`. D-DMPs provide a novel framework for learning and generalizing movement primitives.

## Repository Structure

```
├── README.md
├── dmp
│   ├── __init__.py
│   ├── data_loader.py
│   ├── dmp.py
│   ├── gp.py
│   ├── goal_babbling.py
│   ├── intrinsic_motivation.py
│   ├── plotting.py
│   └── train.py
├── requirements.txt
├── setup.py
```

## Running D-DMPs

### Compute Intention

```python
from dmp.dmp import DistributionalDMP
import torch

train_inputs, train_outputs = load_goal_babbling_trajectories('../Data/Goal Babbling/Pickle Files')

ddmp = DistributionalDMP(input_dim=2, output_dim=2, num_basis=20, basis_stddev=0.5, intention=True)
ddmp.intention = False
initialize_weights(ddmp)
ddmp.learn(train_inputs[0], train_outputs[0])
ddmp.intention = True

# Compute the Intention
goal = [1, 0]
tau_i = 0.3
intentions = ddmp.compute_intention(train_inputs[0])
```


### Rollout DMP

```python
from dmp.dmp import DistributionalDMP
import torch

train_inputs, train_outputs = load_goal_babbling_trajectories('../Data/Goal Babbling/Pickle Files')

ddmp = DistributionalDMP(input_dim=2, output_dim=2, num_basis=20, basis_stddev=0.5, intention=True)
ddmp.learn(train_inputs[0], train_outputs[0])

# Rollout Comparison (D-DMP vs Goal-Conditioned DMP)
x = np.zeros(2)
plt.figure()
plt.plot(ddmp.rollout([1, 0]).T[0], label='D-DMP')
plt.plot(goal_conditioned_dmp.rollout(x, goal).T[0], alpha=0.8, label='GC-DMP')
plt.legend()
```

###  Transfer Learning

```python
from dmp.dmp import DistributionalDMP
import torch

train_inputs, train_outputs = load_goal_babbling_trajectories('../Data/Goal Babbling/Pickle Files')

source = DistributionalDMP(input_dim=2, output_dim=4, num_basis=20, basis_stddev=0.5, intention=True)
source.learn(train_inputs[0], train_outputs[0])
target = DistributionalDMP(input_dim=2, output_dim=1, num_basis=20, basis_stddev=0.5, intention=True)
transfer_learning(source, target)
```


## Installation

To use the system you first need to install the python dependencies. To do this run the command:

`$ pip install -e .`

Alternatively, you can install the dependencies listed in `requirements.txt`.

## Examples

### Goal Babbling

In `goal_babbling_example.py` we give an example of how to generate goal babbling trajectories using the `save_goal_babbling_trajectories` function.

#### Generate Goal Babbling Trajectories

Generating a goal babbling trajectory is as simple as calling the `save_goal_babbling_trajectories` function with the appropriate arguments. This will save the specified number of goal babbling trajectories to pickle files at the desired location.

The following code will save 10 goal babbling trajectories to 10 pickle files at `../Data/Goal Babbling/Pickle Files/`. The trajectories will consist of 100 samples, 200 steps each, 2 dimensions for the output, and with time scaling tau equal to 0.3.

```python
from dmp.goal_babbling import *

if __name__ == '__main__':
    # Save some goal babbling trajectories
    save_goal_babbling_trajectories('../Data/Goal Babbling/Pickle Files', num_trajectories=10, num_samples=100, num_timesteps=200, output_dim=2, tau=0.3)
```

### Intrinsic Motivation

In `intrinsic_motivation.py` we provide the code for generating novel goal babbling trajectories, and for formulating an intrinsic motivation based on the mismatch between the predictions of multiple D-DMPs. In `intrinsic_example.py` we demonstrate how to use these methods.

#### Learn Intentional D-DMPs

When learning a D-DMP model, in order to be able to compute the intention we set `intention=True`, otherwise we set it to `False`. If we set it to `True` then it will learn using intention, otherwise it will learn using the actual goal.

```python
ddmp = DistributionalDMP(input_dim=2, output_dim=2, num_basis=20, basis_stddev=0.5, intention=True)
ddmp.learn(train_inputs[0], train_outputs[0])
ddmp.intention = False
initialize_weights(ddmp)
ddmp.learn(train_inputs[0], train_outputs[0])
ddmp.intention = True
```

### Goal Babbling using Intrinsic Motivation

In `intrinsic_example.py` we show how to learn multiple intentional D-DMP models and to use them to bias the goals selected during goal babbling. We use the mismatch between the predictions of the models to compute the intrinsic motivation. The more the mismatch, the higher the intrinsic motivation.

The following sections demonstrate how to load goal babbling data, how to compute the intrinsic motivation, and how to generate a goal babbling trajectory using intrinsic motivation.

#### Initialization

We initialize the D-DMP models, and we compute the predictions for each model for all the demonstrations in the dataset. These predictions are the same for all demonstrations, since the only thing that differs between them is the goal. Thus we only need to compute the predictions once.

```python
num_models = 10

train_inputs, train_outputs = get_goal_babbling_data()
# Convert to torch tensor
train_inputs = [torch.tensor(i).type(ddmp.dtype) for i in train_inputs]
train_outputs = [torch.tensor(o).type(ddmp.dtype) for o in train_outputs]

# Create the models
models = []
predictions = []
for _ in range(num_models):
    models.append(DistributionalDMP(input_dim=2, output_dim=2, num_basis=20, basis_stddev=0.5, intention=True))
    initialize_weights(models[-1])
    models[-1].learn(train_inputs[0], train_outputs[0])
    models[-1].intention = False
    predictions.append(models[-1].evaluate(train_inputs[0], train_outputs[0]))
```

#### Compute Intrinsic Motivation

Here we compute the intrinsic motivations for each of the models, and we sum them up.

```python
# Compute the intrinsic motivation for each model
intrinsic_motivations = []
for m in models:
    intrinsic_motivations.append(get_intrinsic_motivation(m, train_inputs[0], train_outputs[0], models[:i] + models[i+1:]))
# Sum up the intrinsic motivations
intrinsic_motivation = torch.sum(torch.stack(intrinsic_motivations), dim=0).squeeze()
intrinsic_motivation /= num_models - 1
```

#### Generate Goal Babbling Trajectory

The last step is to generate the goal babbling trajectory. This is done using the learned models and the computed intrinsic motivation.

```python
# Generate the goal babbling trajectory
goals = np.zeros((len(train_inputs[0]), 2))
for i, t in enumerate(range(len(train_inputs[0])-1, 0, -1)):
    for m in models:
        m.intention = False
    prob = torch.nn.functional.softmax(intrinsic_motivation[t] / 0.1, dim=0).detach().numpy() # np.exp(intrinsic_motivation[t])
    goal = np.random.choice([-0.25, 0.25], p=prob)
    goals[i] = [goal, 0]
goals[0] = [0, 0]
```


## Citation

If you use this code in your research, please cite the following paper:

```
@article{distributional-dynamic-movement-primitives},
  author = {Pritom Sarker},
  title = {{Distributional Dynamic Movement Primitives}},
  year = {2020},
  url = {www.github.com/pritoms/Distributional-Dynamic-Movement-Primitives-D-DMPs}
}
```

## License

All the code in this repository is released under the MIT License.
