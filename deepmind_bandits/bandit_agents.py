import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsilonGreedyAlgorithm:
    """
    Epsilon-Greedy bandit algorithm.

    With probability epsilon, explores by choosing a random action.
    Otherwise, exploits by choosing the action with highest estimated value.
    """
    def __init__(self, num_actions, initial_q=0.0, alfa=None):
        assert int(num_actions) >= 1, "num_actions must be >= 1"
        self.K = int(num_actions)
        self.Q = np.full(self.K, float(initial_q), dtype=float)
        self.N = np.zeros(self.K, dtype=int)
        self.epsilon = 0.1
        self.alpha = None if alfa is None else float(alfa)

    def _argmax_random(self, arr):
        """Break ties uniformly at random."""
        ties = np.flatnonzero(arr == arr.max())
        return int(np.random.choice(ties))

    def select_action_greedy(self):
        """
        Select action using epsilon-greedy strategy.
        Uses a Bernoulli draw: explore with probability epsilon, otherwise exploit.
        """
        # Greedy action (random tie-break)
        action_greedy = self._argmax_random(self.Q)

        # Bernoulli trial: 1 = explore, 0 = exploit
        bern_sample = np.random.binomial(n=1, p=self.epsilon, size=1)[0]

        if self.K == 1:
            # Only one arm available
            return 0

        if bern_sample == 1:
            # Explore: choose a random arm that is NOT the greedy one
            choices = [a for a in range(self.K) if a != action_greedy]
            action = int(np.random.choice(choices))
            return action
        else:
            # Exploit: return greedy action
            return int(action_greedy)

    def update_values(self, action, reward, alpha=None):
        """Update action-value estimates using incremental mean."""
        a = int(action)
        r = float(reward)
        self.N[a] += 1

        # Priority: explicit alpha argument -> self.alpha from constructor -> sample-average
        if alpha is not None:
            step_size = float(alpha)
        elif self.alpha is not None:
            step_size = float(self.alpha)
        else:
            step_size = 1.0 / self.N[a]

        self.Q[a] += step_size * (r - self.Q[a])


class UpperConfidenceBound:
    """
    Upper Confidence Bound (UCB) bandit algorithm.

    Balances exploration and exploitation using confidence bounds.
    Actions with higher uncertainty are explored more.
    """
    def __init__(self, num_actions, initial_q=0.0, c=1.0, alfa=None):
        assert int(num_actions) >= 1, "num_actions must be >= 1"
        self.K = int(num_actions)
        self.Q = np.full(self.K, float(initial_q), dtype=float)
        self.U = np.full(self.K, np.inf, dtype=float)  # Infinite uncertainty for untried actions
        self.N = np.zeros(self.K, dtype=int)
        self.c = float(c)
        self.t = 0
        self.alpha = None if alfa is None else float(alfa)

    def _argmax_random(self, arr):
        """Break ties uniformly at random."""
        ties = np.flatnonzero(arr == arr.max())
        return int(np.random.choice(ties))

    def select_action(self):
        """Select action using UCB criterion: argmax(Q + U)."""
        action_greedy = self._argmax_random(self.Q + self.U)
        return action_greedy

    def update_values(self, action, reward, alpha=None):
        """Update action-value estimates and uncertainty bounds."""
        a = int(action)
        r = float(reward)
        self.t += 1
        self.N[a] += 1

        # Update uncertainty bound
        self.U[a] = self.c * np.sqrt(np.log(self.t) / self.N[a])

        # Priority: explicit alpha argument -> self.alpha from constructor -> sample-average
        if alpha is not None:
            step_size = float(alpha)
        elif self.alpha is not None:
            step_size = float(self.alpha)
        else:
            step_size = 1.0 / self.N[a]

        self.Q[a] += step_size * (r - self.Q[a])


class ThompsonSampling:
    """
    Thompson Sampling for Gaussian bandits.

    Uses Bayesian inference with Gaussian posteriors to model uncertainty
    about arm rewards. Samples from posterior distributions to select actions.
    """
    def __init__(self, num_actions, initial_q=0.0, tau0_var=np.inf, mu0=0.0):
        assert int(num_actions) >= 1, "num_actions must be >= 1"
        self.K = int(num_actions)

        # Running sample mean per arm
        self.Q = np.full(self.K, float(initial_q), dtype=float)

        # Posterior parameters per arm: mean and variance of q(a)
        self.means = np.full(self.K, float(mu0), dtype=float)
        self.tau = np.full(self.K, 1.0, dtype=float)  # Start with nonzero variance

        self.N = np.zeros(self.K, dtype=int)

        # Pooled variance (online estimation)
        self.pooled_mean = 0.0
        self.pooled_M2 = 0.0
        self.N_pooled = 0
        self.pooled_var = 1.0  # Keep positive default

        # Prior parameters (variance, not std)
        self.tau0_var = float(tau0_var)  # Use np.inf for flat prior
        self.mu0 = float(mu0)

    def _argmax_random(self, arr):
        """Break ties uniformly at random."""
        ties = np.flatnonzero(arr == arr.max())
        return int(np.random.choice(ties))

    def select_action(self):
        """
        Sample from Beta distributions and select action with highest sample.
        Ensures each arm is pulled at least once.
        """
        # Ensure each arm is pulled at least once to avoid N=0 issues
        unseen = np.flatnonzero(self.N == 0)
        if unseen.size > 0:
            return int(np.random.choice(unseen))

        samples = np.empty(self.K, dtype=float)
        for a in range(self.K):
            samples[a] = np.random.normal(self.means[a], np.sqrt(self.tau[a]))
        return self._argmax_random(samples)

    def pooled_variance_update(self, reward):
        """Update pooled variance estimate using Welford's online algorithm."""
        r = float(reward)
        self.N_pooled += 1
        n = self.N_pooled

        delta = r - self.pooled_mean
        self.pooled_mean += delta / n
        self.pooled_M2 += delta * (r - self.pooled_mean)

        # Sample variance: M2/(n-1)
        if n > 1:
            self.pooled_var = self.pooled_M2 / (n - 1)
        else:
            self.pooled_var = 1.0  # Keep a safe positive value

        return self.pooled_var

    def update_posterior(self, a: int):
        """Update Gaussian posterior parameters for action a."""
        a = int(a)
        n = self.N[a]
        if n <= 0:
            return  # No data yet for this arm

        sigma2 = self.pooled_var if self.pooled_var > 0 else 1.0

        # Flat prior: posterior mean = sample mean, posterior var = sigma2/n
        if np.isinf(self.tau0_var):
            self.means[a] = self.Q[a]
            self.tau[a] = sigma2 / n
            return

        # Conjugate Gaussian update (prior variance = tau0_var)
        tau0_var = self.tau0_var
        post_var = 1.0 / (1.0 / tau0_var + n / sigma2)
        post_mean = post_var * (self.mu0 / tau0_var + n * self.Q[a] / sigma2)

        self.means[a] = post_mean
        self.tau[a] = post_var

    def update_values(self, action, reward):
        """Update posterior distribution parameters."""
        a = int(action)
        r = float(reward)

        # Update pooled variance first (uses this reward)
        self.pooled_variance_update(r)

        # Update sample mean for chosen arm
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]

        # Now update posterior using the updated sample mean
        self.update_posterior(a)


class SimpleLinearModel(nn.Module):
    """
    Simple linear model for gradient bandit algorithm.

    Maps a fixed input to action preferences (logits).
    """
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self):
        x = torch.tensor([1.0])
        # returns: (batch, output_dim) logits
        return self.linear(x)


class GradientBanditAlgorithm:
    """
    Gradient Bandit Algorithm using policy gradients (REINFORCE).

    Uses a neural network to learn action preferences and selects actions
    via softmax policy. Updates using policy gradient with baseline.
    """
    def __init__(self, num_actions, input_dim=1, lr=0.1, use_running_mean=True):
        assert int(num_actions) >= 1, "num_actions must be >= 1"

        self.Q = np.full(num_actions, 0.0, dtype=float)
        self.N = np.zeros((num_actions, 1), dtype=int)
        self.model = SimpleLinearModel(input_dim, num_actions)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.num_actions = num_actions
        self.logits = None
        self.use_running_mean = use_running_mean
        self.running_mean = 0
        self.N_total = 1
        self.loss = None
        self.output_loss = None

    def bandit_loss(self, logits, actions, rewards):
        """Policy gradient loss for bandits."""
        rewards = torch.tensor(rewards, requires_grad=False)
        return rewards * F.cross_entropy(logits, actions)

    def select_action(self):
        """Select action using softmax policy (greedy)."""
        self.logits = self.model()
        action = torch.argmax(self.logits.detach())
        return action

    def update_values(self, action, reward, alpha=None):
        """Update action-value estimates using incremental mean."""
        a = int(action)
        r = float(reward)
        self.N[a] += 1
        self.N_total += 1

        if self.use_running_mean:
            self.running_mean = self.running_mean + (reward - self.running_mean) / self.N_total
            baseline_adjusted_reward = reward - self.running_mean
        else:
            baseline_adjusted_reward = reward

        # Compute loss
        action_tensor = torch.tensor(a, dtype=torch.long)
        self.loss = self.bandit_loss(self.logits, action_tensor, baseline_adjusted_reward)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.output_loss = self.loss.detach()
        self.optimizer.step()

        # Update Q-values
        self.Q[a] += (r - self.Q[a]) / self.N[a]
