# DeepMind Bandits

A comprehensive Python library for multi-armed bandit algorithms with PyTorch integration. This project implements classical and modern bandit algorithms with detailed Jupyter notebook tutorials.

**Based on**: [DeepMind x UCL RL Lecture Series - Exploration & Control [2/13]](https://www.youtube.com/watch?v=aQJP3Z2Ho8U&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)

This implementation covers key sections from the video lecture, providing hands-on implementations of the multi-armed bandit algorithms discussed.

## Features

- **Multiple Bandit Algorithms**:
  - Epsilon-Greedy
  - Upper Confidence Bound (UCB)
  - Thompson Sampling
  - Gradient Bandit (REINFORCE with baseline)

- **Flexible Environment**:
  - Gaussian bandits with configurable means and standard deviations
  - Easy-to-use API for creating custom bandit problems

- **Comprehensive Analysis Tools**:
  - Real-time tracking of Q-values, regret, and rewards
  - Visualization of learning progression
  - Performance metrics and statistics

- **PyTorch Integration**:
  - Neural network-based gradient bandit implementation
  - Policy gradient methods (REINFORCE)
  - Automatic differentiation for optimization

## Installation

```bash
pip install numpy==1.26.4 matplotlib==3.8.4 torch==2.5.1
```

## Quick Start

```python
from deepmind_bandits import (
    GaussianBandits,
    EpsilonGreedyAlgorithm,
    BanditDataAnalyzer
)

# Create environment
means = [1.0, 2.0, -1.0, 0.0]
stds = [0.1, 0.2, 0.1, 0.3]
env = GaussianBandits(means, stds)

# Initialize algorithm
agent = EpsilonGreedyAlgorithm(num_actions=4, initial_q=0.0)
analyzer = BanditDataAnalyzer(means, num_actions=4)

# Training loop
for t in range(1000):
    action = agent.select_action_greedy()
    reward = env.pull_arm(action)
    agent.update_values(action, reward)
    analyzer.update_and_analyze(action, reward)

# Analyze results
analyzer.finalize_analysis()
analyzer.plot_Qvalue()
analyzer.plot_regret()
analyzer.plot_cumulative_reward()
```

## Algorithms

### 1. Epsilon-Greedy

Balances exploration and exploitation with probability epsilon.

```python
from deepmind_bandits import EpsilonGreedyAlgorithm

agent = EpsilonGreedyAlgorithm(
    num_actions=4,
    initial_q=0.0,
    alfa=0.1  # Optional: constant step size
)
```

**Notebook**: `02_epsilon_greedy.ipynb`

### 2. Upper Confidence Bound (UCB)

Uses confidence bounds to balance exploration and exploitation.

```python
from deepmind_bandits import UpperConfidenceBound

agent = UpperConfidenceBound(
    num_actions=4,
    initial_q=0.0,
    c=2.0  # Exploration parameter
)
```

**Notebook**: `04_ucb.ipynb`

### 3. Thompson Sampling

Bayesian approach using posterior distributions.

**Note**: This implementation uses **Gaussian** priors/posteriors (for continuous rewards) rather than Beta-Bernoulli (which is typically used for binary rewards). This is more suitable for the continuous reward setting in our Gaussian bandits environment.

```python
from deepmind_bandits import ThompsonSampling

agent = ThompsonSampling(
    num_actions=4,
    initial_q=0.0,
    tau0_var=np.inf,  # Flat prior (use finite value for informative prior)
    mu0=0.0           # Prior mean
)
```

**Notebook**: `05_thompson_sampling.ipynb`

### 4. Gradient Bandit

Policy gradient method using neural networks.

```python
from deepmind_bandits import GradientBanditAlgorithm

agent = GradientBanditAlgorithm(
    num_actions=4,
    input_dim=1,
    lr=0.1,
    use_running_mean=True  # Use baseline for variance reduction
)

# Training loop
for t in range(1000):
    action = agent.select_action()
    reward = env.pull_arm(action)
    agent.update_values(action, reward)
```

**Notebook**: `03_gradient_bandit.ipynb`

## Project Structure

```
deepmind-bandits/
├── deepmind_bandits/
│   ├── __init__.py
│   ├── bandit_generator.py    # Environment (GaussianBandits)
│   ├── bandit_metrics.py      # Analysis tools (BanditDataAnalyzer)
│   └── bandit_agents.py       # All bandit algorithms
├── 01_intro_bandits.ipynb
├── 02_epsilon_greedy.ipynb
├── 03_gradient_bandit.ipynb
├── 04_ucb.ipynb
├── 05_thompson_sampling.ipynb
└── README.md
```

## Notebooks

### 1. Introduction to Bandits (`01_intro_bandits.ipynb`)
- Multi-armed bandit problem overview
- Environment setup
- Basic concepts

### 2. Epsilon-Greedy (`02_epsilon_greedy.ipynb`)
- Exploration vs exploitation trade-off
- Epsilon-greedy strategy
- Performance analysis

### 3. Gradient Bandit (`03_gradient_bandit.ipynb`)
- Policy gradient methods
- REINFORCE algorithm
- Baseline for variance reduction
- PyTorch implementation

### 4. Upper Confidence Bound (`04_ucb.ipynb`)
- Confidence-based exploration
- UCB algorithm
- Theoretical guarantees

### 5. Thompson Sampling (`05_thompson_sampling.ipynb`)
- Bayesian approach
- Posterior sampling
- Gaussian conjugate priors (not Beta-Bernoulli, due to continuous rewards)

## API Reference

### GaussianBandits

Environment for Gaussian multi-armed bandits.

```python
env = GaussianBandits(means, stds)
reward = env.pull_arm(action)
```

### BanditDataAnalyzer

Tracks and visualizes bandit performance.

```python
analyzer = BanditDataAnalyzer(means, num_actions)
analyzer.update_and_analyze(action, reward, loss_sample=None)
analyzer.finalize_analysis()

# Plotting methods
analyzer.plot_Qvalue()           # Q-value progression
analyzer.plot_regret()           # Cumulative regret
analyzer.plot_cumulative_reward() # Cumulative rewards
analyzer.loss_hist()             # Loss histogram (if available)
```

## Key Concepts

### Multi-Armed Bandit Problem

The agent must choose between K actions (arms) to maximize cumulative reward over time without knowing the true reward distributions.

### Exploration vs Exploitation

- **Exploitation**: Choose the action with highest estimated value
- **Exploration**: Try other actions to improve estimates

### Regret

Regret measures the difference between the optimal policy and the agent's performance:

```
Regret = Σ(optimal_reward - received_reward)
```

### Baseline in Gradient Bandits

The baseline reduces variance in policy gradient estimates without adding bias:

```python
gradient ∝ (reward - baseline) * ∇log(π(action))
```

## Mathematical Foundations

### REINFORCE Update Rule

For gradient bandits:

```
θ ← θ + α(R - baseline)∇log(π_θ(A))
```

Where:
- `θ`: Policy parameters
- `α`: Learning rate
- `R`: Observed reward
- `baseline`: Running mean of rewards
- `π_θ(A)`: Probability of action A

### UCB Selection

```
A = argmax[Q(a) + c√(ln(t)/N(a))]
```

Where:
- `Q(a)`: Estimated value of action a
- `c`: Exploration parameter
- `t`: Total time steps
- `N(a)`: Times action a was selected

## Examples

See the Jupyter notebooks for detailed examples and experiments.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Acknowledgments

Inspired by DeepMind's research on reinforcement learning and multi-armed bandits.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another
