import networkx as nx
from ..core.metricssuite import MetricsSuite


def naive_optimizer(M_input: MetricsSuite, inplace=False):
    """Naive optimizer for the graph layout.

    This optimizer generates layouts based on four NetworkX graph layout algorithms:
    - spring_layout
    - shell_layout
    - circular_layout
    - kamada_kawai_layout
    Then chooses the layout that maximizes the MetricSuite score.

    Parameters
    ----------
    M : MetricSuite
        The MetricSuite object to be optimized.
    inplace : bool, optional
        Whether to modify the MetricSuite object in place or return a new one. Default is False.

    Returns
    -------
    MetricSuite
        The optimized MetricSuite object.
    """
    layouts = {
        "spring": nx.spring_layout,
        "shell": nx.shell_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }

    scores = {}
    poss = {}
    for layout_name, layout_func in layouts.items():
        M = M_input.copy() if not inplace else M_input
        poss[layout_name] = layout_func(M.G)
        M.apply_layout(poss[layout_name])
        M.calculate_metrics()
        scores[layout_name] = M.combine_metrics()

    # Choose the best layout
    best_layout = max(scores, key=scores.get)
    M = M_input.copy() if not inplace else M_input
    M.apply_layout(poss[best_layout])

    print(f"Best layout: {best_layout}. Score: {scores[best_layout]}")
    return M


def optimize(M_input: MetricsSuite, inplace=False):
    """Default optimizer for the graph layout."""
    return naive_optimizer(M_input, inplace=inplace)
