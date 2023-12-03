import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

winning_streak_length = 45

def count_sequences(my_array, ones_in_a_row):
    """
    Count the number of sequences with a specified number of consecutive ones in a NumPy array.

    Parameters:
    - my_array (numpy.ndarray): The input NumPy array.
    - ones_in_a_row (int): The desired number of consecutive ones in a sequence.

    Returns:
    - int: The count of sequences with the specified number of consecutive ones.

    Example:
    >>> my_array = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
    >>> count_sequences(my_array, ones_in_a_row=3)
    2
    """
    seq = np.ones(ones_in_a_row, dtype=int)
    count = 0

    i = 0
    while i < len(my_array):
        if np.array_equal(my_array[i : i + len(seq)], seq):
            count += 1
            my_array = np.delete(my_array, range(i, i + len(seq)))
        else:
            i += 1

    return count

def run_simulations(num_experiments=10, my_probs=[0.85, 0.92, 0.95, 0.97]):
    """
    Run simulations to investigate winning streaks.

    Parameters:
    - num_experiments (int): Number of simulation experiments to run.

    Returns:
    - df_games (pd.DataFrame): DataFrame containing simulation results for each experiment.
    """
    streaks = []
    probs = []
    n_games = []
    exp_no = []

    for experiment in range(num_experiments):
        for n in np.arange(100, 1100, 100):
            for p in my_probs:
                games_results = np.random.binomial(n=1, p=p, size=n)
                streak = count_sequences(games_results, winning_streak_length)

                probs.append(p)
                streaks.append(streak)
                n_games.append(n)
                exp_no.append(experiment)

    df_games = pd.DataFrame({"exp_no": exp_no, "streak": streaks, "prob": probs, "n_games": n_games})
    df_games["prob"] = df_games.prob.astype("category")

    return df_games

def visualize_results(df_games, confidence_interval=99):
    """
    Visualize the results of winning streak simulations.

    Parameters:
    - df_games (pd.DataFrame): DataFrame containing simulation results.
    - confidence_interval (int, optional): Confidence interval for error bars (default is 99).
    """
    plt.figure(dpi=200)
    sns.set_style("whitegrid")

    sns.lineplot(
        x="n_games",
        y="streak",
        hue="prob",
        data=df_games,
        errorbar=("ci", confidence_interval),
        linewidth=1,
    )

    plt.ylabel(f"Number of {winning_streak_length}-games winning streaks")
    plt.xlabel("Number of Games Played", labelpad=10)

    plt.xticks(range(100, 1001, 100))
    plt.yticks(range(0, 11, 1))

    plt.title(f"Winning Streaks of Length {winning_streak_length} vs Games Played")

    plt.legend(
        title="Probability of\nwinning in a \nsingle game",
        bbox_to_anchor=(1, 0.5),
        loc="center left",
        frameon=False,
        fontsize="small",
        title_fontsize="small",
    )

    plt.annotate(
        f"Number of Simulations Conducted: {num_experiments}\nConfidence level: {confidence_interval}",
        xy=(0.5, -0.2),
        xycoords="axes fraction",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", alpha=0.1),
        fontsize=7,
        color="gray",
    )

    plt.show()

if __name__ == "__main__":
    num_experiments = 1000  # Set the desired number of experiments
    df_games = run_simulations(num_experiments=num_experiments, my_probs=[0.85, 0.90, 0.92, 0.95, 0.97])
    visualize_results(df_games)
