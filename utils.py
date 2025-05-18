import matplotlib.pyplot as plt
from typing import Dict, List

from agents import RLAgent


def plot_comparison(
    results_list: List[Dict], metric: str = "win_rate", title: str = None
):
    """
    Compare multiple training results.

    Args:
        results_list: List of training results
        metric: Metric to compare
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    for results in results_list:
        agent_name = results["agent_name"]
        metrics = results["metrics"]
        episodes = results["episodes"]

        if metric in metrics and len(metrics[metric]) > 0:
            eval_points = range(0, episodes, episodes // len(metrics[metric]))
            plt.plot(eval_points, metrics[metric], label=agent_name)

    plt.xlabel("Episodes")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title or f"Comparison of {metric.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_action_distribution(agent: RLAgent):
    """
    Plot distribution of actions taken.

    Args:
        agent: Agent to analyze
    """
    action_dist = agent.metrics["action_dist"]

    plt.figure(figsize=(10, 6))
    x = range(len(action_dist))
    plt.bar(x, action_dist)

    # Add grid coordinates as x-tick labels
    n = agent.env.n
    labels = [f"({a//n}, {a%n})" for a in x]
    plt.xticks(x, labels, rotation=45)

    plt.xlabel("Action (x, y)")
    plt.ylabel("Frequency")
    plt.title(f"Action Distribution for {agent.name}")
    plt.tight_layout()
    plt.show()
