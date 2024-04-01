import unittest
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import src as gr


class TestGraphReadability(unittest.TestCase):

    def setUp(self):
        self.graphs = []
        self.graph_names = []
        graphs = os.walk("../graphs")
        for root, dirs, files in graphs:
            for file in files:
                if file.endswith(".gml"):
                    self.graphs.append(nx.read_gml(os.path.join(root, file)))
                elif file.endswith(".graphml"):
                    self.graphs.append(nx.read_graphml(os.path.join(root, file)))
                self.graph_names.append(file)

    def test_graphs_loading(self):
        self.assertTrue(len(self.graphs) > 0, "No graphs loaded")

    def test_metrics_calculation(self):
        for G in self.graphs:
            M = gr.MetricsSuite(G)
            M.calculate_metrics()
            self.assertIsNotNone(M._graph, "Graph object is None after calculation")
            self.assertTrue(len(M.metric_table()) > 0, "No metrics calculated")

    def test_plotting(self):
        for G, name in zip(self.graphs, self.graph_names):
            M = gr.MetricsSuite(G)
            M.calculate_metrics()
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            plt.suptitle(name)
            metric_table = pd.Series(M.metric_table())
            ax[1].bar(metric_table.index, metric_table.values)
            ax[1].tick_params(axis="x", rotation=90)
            gr.draw_graph(M._graph, ax=ax[0])
            plt.close(fig)  # Ensure that the figure is closed after plotting

    def test_dataframe_generation(self):
        metric_tables = []
        for G in self.graphs:
            M = gr.MetricsSuite(G)
            M.calculate_metrics()
            metric_table = pd.Series(M.metric_table())
            metric_tables.append(metric_table)
        tables = pd.DataFrame.from_records(
            metric_tables, index=self.graph_names, columns=metric_table.index
        ).sort_values(by="Combined", ascending=False)
        self.assertTrue(
            len(tables) == len(self.graphs),
            "Mismatch in number of graphs and metric tables",
        )

    def test_barplot(self):
        metric_tables = []
        for G in self.graphs:
            M = gr.MetricsSuite(G)
            M.calculate_metrics()
            metric_table = pd.Series(M.metric_table())
            metric_tables.append(metric_table)
        tables = pd.DataFrame.from_records(
            metric_tables, index=self.graph_names, columns=metric_table.index
        ).sort_values(by="Combined", ascending=False)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        gs = axs[1, 1].get_gridspec()
        for ax in axs[1, :]:
            ax.remove()
        ax_bottom = fig.add_subplot(gs[1, :])
        tables.iloc[[3, 0]].T.plot(
            ax=ax_bottom, kind="barh", stacked=False, color=["r", "b"], legend=False
        )
        plt.close(fig)  # Ensure that the figure is closed after plotting


if __name__ == "__main__":
    unittest.main()
