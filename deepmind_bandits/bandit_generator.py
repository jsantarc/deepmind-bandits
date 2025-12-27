import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class GaussianBandits:
    def __init__(self, means, stds):
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.num_arms = len(means)
        self.v_star = np.max(self.means)
        self.q_a = self.means

    
    def pull_arm(self, arm_index):



        return np.random.normal(self.means[arm_index], self.stds[arm_index])

    def plot_parametric_distributions(self, arm_indices=None, x_range=None, num_points=1000,
                                      ax=None, orientation="vertical", fill=False,
                                      invert_x=False, show=True):
        """
        Plot parametric Gaussian PDFs for selected arms.

        Parameters:
        -----------
        arm_indices : list, optional
            Indices of arms to plot
        x_range : tuple, optional
            (min, max) range for x-axis
        num_points : int
            Number of points for PDF calculation
        ax : matplotlib axis, optional
            Axis to plot on (creates new figure if None)
        orientation : str
            "vertical" or "horizontal" orientation
        fill : bool
            Whether to fill under the curves
        invert_x : bool
            Whether to invert x-axis (for horizontal orientation)
        show : bool
            Whether to call plt.show()
        """
        if arm_indices is None:
            arm_indices = range(self.num_arms)

        # Automatically choose a reasonable x-range
        if x_range is None:
            min_x = np.min(self.means - 4 * self.stds)
            max_x = np.max(self.means + 4 * self.stds)
            x = np.linspace(min_x, max_x, num_points)
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)

        # Create figure if no axis provided
        if ax is None:
            plt.figure()
            ax = plt.gca()

        for arm in arm_indices:
            mu = self.means[arm]
            sigma = self.stds[arm]

            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            if orientation == "horizontal":
                if fill:
                    ax.fill_betweenx(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (μ={mu:.2f}, σ={sigma:.2f})")
                else:
                    ax.plot(pdf, x, label=f"Arm {arm} (μ={mu:.2f}, σ={sigma:.2f})")
                ax.set_ylabel("Reward")
                ax.set_xlabel("Probability Density")
                if invert_x:
                    ax.invert_xaxis()
            else:
                if fill:
                    ax.fill_between(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (μ={mu:.2f}, σ={sigma:.2f})")
                else:
                    ax.plot(x, pdf, label=f"Arm {arm} (μ={mu:.2f}, σ={sigma:.2f})")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Probability Density")

        if orientation == "vertical":
            ax.set_title("Gaussian Bandit Reward Distributions")
        ax.legend(loc='best')

        if show:
            plt.show()

    def plot_time_series_with_distribution(self, T=200, arm_indices=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if arm_indices is None:
            arm_indices = list(range(self.num_arms))

        num_arms = len(arm_indices)

        # ---- time series: (num_arms, T) ----
        time_series = np.zeros((num_arms, T))
        for n, arm in enumerate(arm_indices):
            time_series[n, :] = np.random.normal(
                self.means[arm],
                self.stds[arm],
                size=T
            )

        # ---- shared reward range ----
        y_min = np.min(self.means - 4 * self.stds)
        y_max = np.max(self.means + 4 * self.stds)

        # ---- figure & axes ----
        fig, (ax_ts, ax_dist) = plt.subplots(
            1, 2, figsize=(11, 6),
            gridspec_kw={"width_ratios": [4.0, 1.2]},
            sharey=True
        )

        # ---- left: time series ----
        ax_ts.plot(time_series.T)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Reward")
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.legend([f"Arm {a}" for a in arm_indices])
        ax_ts.grid(True, alpha=0.25)

        # ---- right: distribution ----
        self.plot_parametric_distributions(
            arm_indices=arm_indices,
            x_range=(y_min, y_max),
            ax=ax_dist,
            orientation="horizontal",
            fill=True,
            invert_x=False,
            show=False
        )

        ax_dist.set_yticklabels([])
        ax_dist.grid(True, alpha=0.25)

        fig.suptitle("Time Series Samples with Parametric Distributions", y=0.98)
        plt.show()


class UniformBandits:
    def __init__(self, lows, highs):
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.num_arms = len(lows)
        self.means = (self.lows + self.highs) / 2
        self.v_star = np.max(self.means)
        self.q_a = self.means

    def pull_arm(self, arm_index):
        return np.random.uniform(self.lows[arm_index], self.highs[arm_index])

    def plot_parametric_distributions(self, arm_indices=None, x_range=None, num_points=1000,
                                      ax=None, orientation="vertical", fill=False,
                                      invert_x=False, show=True):
        if arm_indices is None:
            arm_indices = range(self.num_arms)

        if x_range is None:
            min_x = np.min(self.lows) - 0.5
            max_x = np.max(self.highs) + 0.5
            x = np.linspace(min_x, max_x, num_points)
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        for arm in arm_indices:
            low = self.lows[arm]
            high = self.highs[arm]

            pdf = np.where((x >= low) & (x <= high), 1.0 / (high - low), 0)

            if orientation == "horizontal":
                if fill:
                    ax.fill_betweenx(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (low={low:.2f}, high={high:.2f})")
                else:
                    ax.plot(pdf, x, label=f"Arm {arm} (low={low:.2f}, high={high:.2f})")
                ax.set_ylabel("Reward")
                ax.set_xlabel("Probability Density")
                if invert_x:
                    ax.invert_xaxis()
            else:
                if fill:
                    ax.fill_between(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (low={low:.2f}, high={high:.2f})")
                else:
                    ax.plot(x, pdf, label=f"Arm {arm} (low={low:.2f}, high={high:.2f})")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Probability Density")

        if orientation == "vertical":
            ax.set_title("Uniform Bandit Reward Distributions")
        ax.legend(loc='best')

        if show:
            plt.show()

    def plot_time_series_with_distribution(self, T=200, arm_indices=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if arm_indices is None:
            arm_indices = list(range(self.num_arms))

        num_arms = len(arm_indices)

        time_series = np.zeros((num_arms, T))
        for n, arm in enumerate(arm_indices):
            time_series[n, :] = np.random.uniform(
                self.lows[arm],
                self.highs[arm],
                size=T
            )

        y_min = np.min(self.lows) - 0.5
        y_max = np.max(self.highs) + 0.5

        fig, (ax_ts, ax_dist) = plt.subplots(
            1, 2, figsize=(11, 6),
            gridspec_kw={"width_ratios": [4.0, 1.2]},
            sharey=True
        )

        ax_ts.plot(time_series.T)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Reward")
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.legend([f"Arm {a}" for a in arm_indices])
        ax_ts.grid(True, alpha=0.25)

        self.plot_parametric_distributions(
            arm_indices=arm_indices,
            x_range=(y_min, y_max),
            ax=ax_dist,
            orientation="horizontal",
            fill=True,
            invert_x=False,
            show=False
        )

        ax_dist.set_yticklabels([])
        ax_dist.grid(True, alpha=0.25)

        fig.suptitle("Time Series Samples with Parametric Distributions", y=0.98)
        plt.show()


class ExponentialBandits:
    def __init__(self, scales):
        self.scales = np.array(scales)
        self.num_arms = len(scales)
        self.means = self.scales
        self.v_star = np.max(self.means)
        self.q_a = self.means

    def pull_arm(self, arm_index):
        return np.random.exponential(self.scales[arm_index])

    def plot_parametric_distributions(self, arm_indices=None, x_range=None, num_points=1000,
                                      ax=None, orientation="vertical", fill=False,
                                      invert_x=False, show=True):
        if arm_indices is None:
            arm_indices = range(self.num_arms)

        if x_range is None:
            min_x = 0
            max_x = np.max(self.scales) * 4
            x = np.linspace(min_x, max_x, num_points)
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        for arm in arm_indices:
            scale = self.scales[arm]

            pdf = (1 / scale) * np.exp(-x / scale)

            if orientation == "horizontal":
                if fill:
                    ax.fill_betweenx(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (scale={scale:.2f})")
                else:
                    ax.plot(pdf, x, label=f"Arm {arm} (scale={scale:.2f})")
                ax.set_ylabel("Reward")
                ax.set_xlabel("Probability Density")
                if invert_x:
                    ax.invert_xaxis()
            else:
                if fill:
                    ax.fill_between(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (scale={scale:.2f})")
                else:
                    ax.plot(x, pdf, label=f"Arm {arm} (scale={scale:.2f})")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Probability Density")

        if orientation == "vertical":
            ax.set_title("Exponential Bandit Reward Distributions")
        ax.legend(loc='best')

        if show:
            plt.show()

    def plot_time_series_with_distribution(self, T=200, arm_indices=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if arm_indices is None:
            arm_indices = list(range(self.num_arms))

        num_arms = len(arm_indices)

        time_series = np.zeros((num_arms, T))
        for n, arm in enumerate(arm_indices):
            time_series[n, :] = np.random.exponential(
                self.scales[arm],
                size=T
            )

        y_min = 0
        y_max = np.max(self.scales) * 4

        fig, (ax_ts, ax_dist) = plt.subplots(
            1, 2, figsize=(11, 6),
            gridspec_kw={"width_ratios": [4.0, 1.2]},
            sharey=True
        )

        ax_ts.plot(time_series.T)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Reward")
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.legend([f"Arm {a}" for a in arm_indices])
        ax_ts.grid(True, alpha=0.25)

        self.plot_parametric_distributions(
            arm_indices=arm_indices,
            x_range=(y_min, y_max),
            ax=ax_dist,
            orientation="horizontal",
            fill=True,
            invert_x=False,
            show=False
        )

        ax_dist.set_yticklabels([])
        ax_dist.grid(True, alpha=0.25)

        fig.suptitle("Time Series Samples with Parametric Distributions", y=0.98)
        plt.show()


class BetaBandits:
    def __init__(self, alphas, betas):
        self.alphas = np.array(alphas)
        self.betas = np.array(betas)
        self.num_arms = len(alphas)
        self.means = self.alphas / (self.alphas + self.betas)
        self.v_star = np.max(self.means)
        self.q_a = self.means

    def pull_arm(self, arm_index):
        return np.random.beta(self.alphas[arm_index], self.betas[arm_index])

    def plot_parametric_distributions(self, arm_indices=None, x_range=None, num_points=1000,
                                      ax=None, orientation="vertical", fill=False,
                                      invert_x=False, show=True):
        if arm_indices is None:
            arm_indices = range(self.num_arms)

        if x_range is None:
            min_x = 0
            max_x = 1
            x = np.linspace(min_x, max_x, num_points)
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        for arm in arm_indices:
            alpha = self.alphas[arm]
            beta = self.betas[arm]

            pdf = stats.beta.pdf(x, alpha, beta)

            if orientation == "horizontal":
                if fill:
                    ax.fill_betweenx(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (α={alpha:.2f}, β={beta:.2f})")
                else:
                    ax.plot(pdf, x, label=f"Arm {arm} (α={alpha:.2f}, β={beta:.2f})")
                ax.set_ylabel("Reward")
                ax.set_xlabel("Probability Density")
                if invert_x:
                    ax.invert_xaxis()
            else:
                if fill:
                    ax.fill_between(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (α={alpha:.2f}, β={beta:.2f})")
                else:
                    ax.plot(x, pdf, label=f"Arm {arm} (α={alpha:.2f}, β={beta:.2f})")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Probability Density")

        if orientation == "vertical":
            ax.set_title("Beta Bandit Reward Distributions")
        ax.legend(loc='best')

        if show:
            plt.show()

    def plot_time_series_with_distribution(self, T=200, arm_indices=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if arm_indices is None:
            arm_indices = list(range(self.num_arms))

        num_arms = len(arm_indices)

        time_series = np.zeros((num_arms, T))
        for n, arm in enumerate(arm_indices):
            time_series[n, :] = np.random.beta(
                self.alphas[arm],
                self.betas[arm],
                size=T
            )

        y_min = 0
        y_max = 1

        fig, (ax_ts, ax_dist) = plt.subplots(
            1, 2, figsize=(11, 6),
            gridspec_kw={"width_ratios": [4.0, 1.2]},
            sharey=True
        )

        ax_ts.plot(time_series.T)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Reward")
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.legend([f"Arm {a}" for a in arm_indices])
        ax_ts.grid(True, alpha=0.25)

        self.plot_parametric_distributions(
            arm_indices=arm_indices,
            x_range=(y_min, y_max),
            ax=ax_dist,
            orientation="horizontal",
            fill=True,
            invert_x=False,
            show=False
        )

        ax_dist.set_yticklabels([])
        ax_dist.grid(True, alpha=0.25)

        fig.suptitle("Time Series Samples with Parametric Distributions", y=0.98)
        plt.show()


class GammaBandits:
    def __init__(self, shapes, scales):
        self.shapes = np.array(shapes)
        self.scales = np.array(scales)
        self.num_arms = len(shapes)
        self.means = self.shapes * self.scales
        self.v_star = np.max(self.means)
        self.q_a = self.means

    def pull_arm(self, arm_index):
        return np.random.gamma(self.shapes[arm_index], self.scales[arm_index])

    def plot_parametric_distributions(self, arm_indices=None, x_range=None, num_points=1000,
                                      ax=None, orientation="vertical", fill=False,
                                      invert_x=False, show=True):
        if arm_indices is None:
            arm_indices = range(self.num_arms)

        if x_range is None:
            min_x = 0
            max_x = np.max(self.means) * 3
            x = np.linspace(min_x, max_x, num_points)
        else:
            x = np.linspace(x_range[0], x_range[1], num_points)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        for arm in arm_indices:
            shape = self.shapes[arm]
            scale = self.scales[arm]

            pdf = stats.gamma.pdf(x, shape, scale=scale)

            if orientation == "horizontal":
                if fill:
                    ax.fill_betweenx(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (k={shape:.2f}, θ={scale:.2f})")
                else:
                    ax.plot(pdf, x, label=f"Arm {arm} (k={shape:.2f}, θ={scale:.2f})")
                ax.set_ylabel("Reward")
                ax.set_xlabel("Probability Density")
                if invert_x:
                    ax.invert_xaxis()
            else:
                if fill:
                    ax.fill_between(x, 0, pdf, alpha=0.3, label=f"Arm {arm} (k={shape:.2f}, θ={scale:.2f})")
                else:
                    ax.plot(x, pdf, label=f"Arm {arm} (k={shape:.2f}, θ={scale:.2f})")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Probability Density")

        if orientation == "vertical":
            ax.set_title("Gamma Bandit Reward Distributions")
        ax.legend(loc='best')

        if show:
            plt.show()

    def plot_time_series_with_distribution(self, T=200, arm_indices=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if arm_indices is None:
            arm_indices = list(range(self.num_arms))

        num_arms = len(arm_indices)

        time_series = np.zeros((num_arms, T))
        for n, arm in enumerate(arm_indices):
            time_series[n, :] = np.random.gamma(
                self.shapes[arm],
                self.scales[arm],
                size=T
            )

        y_min = 0
        y_max = np.max(self.means) * 3

        fig, (ax_ts, ax_dist) = plt.subplots(
            1, 2, figsize=(11, 6),
            gridspec_kw={"width_ratios": [4.0, 1.2]},
            sharey=True
        )

        ax_ts.plot(time_series.T)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Reward")
        ax_ts.set_ylim(y_min, y_max)
        ax_ts.legend([f"Arm {a}" for a in arm_indices])
        ax_ts.grid(True, alpha=0.25)

        self.plot_parametric_distributions(
            arm_indices=arm_indices,
            x_range=(y_min, y_max),
            ax=ax_dist,
            orientation="horizontal",
            fill=True,
            invert_x=False,
            show=False
        )

        ax_dist.set_yticklabels([])
        ax_dist.grid(True, alpha=0.25)

        fig.suptitle("Time Series Samples with Parametric Distributions", y=0.98)
        plt.show()
