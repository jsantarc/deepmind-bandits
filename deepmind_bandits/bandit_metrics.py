import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class BanditDataAnalyzer:
    """
    Analyzes and plots recorded bandit simulation data: Q-values,
    cumulative regret, and cumulative reward (total and per-action).
    """
    def __init__(self, means, num_actions):
        """
        Initializes the analyzer with true environment means and tracking lists.
        """
        self.means = np.array(means)
        self.num_actions = num_actions
        self.optimal_value = np.max(self.means)

        # Tracking lists
        self.Q_t_list = [[] for _ in range(self.num_actions)]
        self.regret_list = []
        self.action_list = []
        self.loss=[]


        # Calculated results
        self.regret = None
        self.Qt = None
        self.cumulative_reward_total = None # Total cumulative reward (your original logic)
        self.cumulative_reward_per_action = None # NEW: Cumulative reward for each action
        self.loss_flag=None

    def update_and_analyze(self, action, reward,loss_sample=None):
        """
        Records a single action-reward step and updates the tracking lists.
        """
        # 1. Update Q_t_list (stores the reward received for this action at this time step)
        if action < self.num_actions:
            self.Q_t_list[action].append(reward)
            self.action_list.append(action)

        # 2. Update Regret list
        immediate_regret = self.optimal_value - reward
        self.regret_list.append(immediate_regret)
        if loss_sample is not None:
            self.loss_flag=True
            loss_dummy=loss_sample.detach().item()
            self.loss.append(loss_dummy)

    def finalize_analysis(self):
        """
        Calculates cumulative metrics and prepares the Q_t array for plotting.
        """
        if not self.regret_list:
            print("Error: No data recorded. Call update_and_analyze() first.")
            return

        # 1. Calculate Cumulative Regret
        self.regret = np.cumsum(self.regret_list)

        # 2. Prepare Padded Q_t Array (Qt)
        max_length = len(self.regret_list) # Total number of steps

        Qt_padded = []
        # Zero-pad shorter lists for per-action cumulative reward calculation
        Qt_raw = [[] for _ in range(self.num_actions)]
        for t, action in enumerate(self.regret_list): # Iterate over total steps
            for a in range(self.num_actions):
                # Check if this action was taken at this step t
                # Since Q_t_list only stores rewards when action was pulled, we need a better structure.
                # Rebuilding the full reward history array (Action vs Time)
                reward_at_step_t = 0
                if t < len(self.Q_t_list[a]):
                    reward_at_step_t = self.Q_t_list[a][t] # This is wrong logic, Q_t_list is not time-aligned

        # --- REVISED LOGIC FOR Q_t and PER-ACTION REWARD ---
        # The Q_t_list needs to be restructured based on when the action was taken.
        # Since we don't store *which* action was taken at *which* time, we must pad
        # based on the assumption that a reward list for an arm is a sequence of *successful pulls*.

        # We will keep your original Qt logic for total reward and use a new structure
        # for per-action cumulative reward by padding with zeros for unpulled steps.

        # Padded Q_t array (Your original logic)
        Qt_padded_for_total = []
        for q_t in self.Q_t_list:
            if not q_t:
                padded_q_t = [0] * max_length
            else:
                # Padding with the last received reward (for Q-value visualization)
                padded_q_t = q_t + [q_t[-1]] * (max_length - len(q_t))
            Qt_padded_for_total.append(padded_q_t)

        self.Qt = np.array(Qt_padded_for_total).T # Q-value visualization array

        # 3. Calculate Cumulative Reward (Total)
        self.cumulative_reward_total = np.cumsum(self.Qt.sum(axis=1))

        # 4. Calculate Cumulative Reward (Per-Action)
        # To accurately get per-action cumulative reward, we must know which action was taken at each time step.
        # Since the original structure doesn't store this, we'll use a **zero-padding approach**
        # that assumes the recorded rewards in Q_t_list are sequentially correct for that arm.

        Qt_zero_padded = []
        for q_t in self.Q_t_list:
            if not q_t:
                padded_q_t = [0] * max_length
            else:
                # Padding with zero for unpulled steps (for reward summation)
                padded_q_t = q_t + [0] * (max_length - len(q_t))
            Qt_zero_padded.append(padded_q_t)

        Qt_zero_array = np.array(Qt_zero_padded).T
        self.cumulative_reward_per_action = np.cumsum(Qt_zero_array, axis=0)


# --- Plotting Methods ---

    def plot_Qvalue(self):
        """Plots the Q-value progression with action transitions shown as arrows."""
        if self.Qt is None:
            return print("Error: Run finalize_analysis() first.")

        # Prepare Q_switch array (Q-values aligned with time steps)
        T = len(self.action_list)
        action_count = [0 for _ in range(self.num_actions)]
        Q_switch = np.ones((T, self.num_actions)) * self.means.reshape(1, -1)
        switch = []
        a_flag = self.action_list[0]

        for i, a in enumerate(self.action_list):
            action_count[a] += 1
            Q_switch[i, a] = self.Q_t_list[a][action_count[a] - 1]
            if a_flag != a:
                switch.append((a_flag, a))
                a_flag = a
            else:
                switch.append(0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot Q-values for all actions as thin lines
        for action in range(self.num_actions):
            ax.plot(Q_switch[:, action], label=f'Action {action}',
                    alpha=0.3, linewidth=1.5, color=f'C{action}')
            # True means as dashed lines
            ax.axhline(y=self.means[action], color=f'C{action}',
                       linestyle='--', alpha=0.2, linewidth=1)

        # Highlight the selected action's Q-value with bold line
        for i, action in enumerate(self.action_list):
            if i < len(self.action_list) - 1:
                ax.plot([i, i+1], [Q_switch[i, action], Q_switch[i+1, action]],
                        color=f'C{action}', linewidth=3, alpha=0.9)

        # Add arrows for transitions
        for i, sw in enumerate(switch):
            if sw != 0:  # There's a switch
                from_action, to_action = sw

                # Get positions for arrow
                x_pos = i
                y_from = Q_switch[i-1, from_action] if i > 0 else Q_switch[i, from_action]
                y_to = Q_switch[i, to_action]

                # Add arrow
                arrow = FancyArrowPatch(
                    (x_pos, y_from), (x_pos, y_to),
                    arrowstyle='->', mutation_scale=25,
                    linewidth=2.5, color='red', alpha=0.7,
                    zorder=10
                )
                ax.add_patch(arrow)

                # Add label
                mid_y = (y_from + y_to) / 2
                ax.text(x_pos + T*0.01, mid_y, f'{from_action}→{to_action}',
                        fontsize=10, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='red', alpha=0.8))

        ax.set_xlabel('Time Steps (T)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Q-value Estimate', fontsize=13, fontweight='bold')
        ax.set_title('Multi-Armed Bandit: Q-values Over Time with Action Transitions',
                     fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print transition summary
        print(f"\nTransitions detected: {len([s for s in switch if s != 0])}")
        for i, sw in enumerate(switch):
            if sw != 0:
                print(f"  Step {i}: Action {sw[0]} → Action {sw[1]}")

    def plot_regret(self):
        """Plots the Cumulative Regret over time."""
        if self.regret is None: return print("Error: Run finalize_analysis() first.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.regret, label='Cumulative Regret', color='red')
        plt.xlabel('Time Steps (T)')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regret Over Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

        if self.loss_flag:
          plt.figure(figsize=(10, 6))
          plt.plot(self.loss, label='Loss', color='blue')
          plt.xlabel('Time Steps (T)')
          plt.ylabel('Loss')
          plt.title('Loss Over Time')
          plt.legend(loc='best')
          plt.grid(True)
          plt.show()

    def plot_cumulative_reward(self):
        """
        Plots the Cumulative Reward (Total and Per-Action) over time.
        """
        if self.cumulative_reward_total is None or self.cumulative_reward_per_action is None:
            return print("Error: Run finalize_analysis() first.")

        plt.figure(figsize=(10, 6))

        # Plot Per-Action Cumulative Rewards
        for i in range(self.num_actions):
            plt.plot(self.cumulative_reward_per_action[:, i],
                     linestyle='--', label=f'Arm {i} Cumulative Reward')

        # Plot Total Cumulative Reward
        plt.plot(self.cumulative_reward_total, label='Total Cumulative Reward',
                 color='black', linewidth=3)

        plt.xlabel('Time Steps (T)')
        plt.ylabel('Cumulative Reward')
        plt.title('Total and Per-Action Cumulative Reward Over Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def loss_hist(self):
        if not self.loss_flag:
            print("Error: No loss data recorded.")
            return None
        
        try:

            plt.figure()
            plt.hist(self.loss, bins=30)
            plt.xlabel("Loss value")
            plt.ylabel("Frequency")
            plt.title("Loss Histogram")
            plt.show()
            loss = np.array(self.loss)

            stats = {
            "mean": loss.mean(),
            "median": np.median(loss),
            "std": loss.std(),
            "min": loss.min(),
            "max": loss.max(),
            "p25": np.percentile(loss, 25),
            "p75": np.percentile(loss, 75),
            "p95": np.percentile(loss, 95)}

            return stats
        except Exception as e:
            print(f"Error generating loss histogram: {e}")
            return None
    