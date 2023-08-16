from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout

from defrag.tests.test_defrag_dashboard import test_dashboard
import sys
import networkx as nx
import matplotlib.pyplot as plt
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.envs.defragmentation_env import OldestFirst
import copy
import numpy as np
import itertools

from random import randint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):
    """
    Responsibilities:
    - call functions
    - all plotting
    """

    def __init__(self, env, agent, tapi_client):
        super().__init__()
        self.env = env
        self.agent = agent
        self.tapi_client = tapi_client
        self.w = None  # No external window yet.

        self.setWindowTitle("DefragDashboard")
        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        pagelayout.addLayout(button_layout)
        self.btn_advance = QPushButton("advance ")
        self.btn_advance.pressed.connect(self.advance_arrivals)
        button_layout.addWidget(self.btn_advance)

        self.btn_drl = QPushButton("start DRL-based defragmentation")
        self.btn_drl.pressed.connect(self.start_drl)
        button_layout.addWidget(self.btn_drl)

        self.btn_reset = QPushButton("reset")
        self.btn_drl.pressed.connect(self.reset_env)
        button_layout.addWidget(self.btn_reset)

        self.btn_topology = QPushButton("show topology")
        self.btn_topology.pressed.connect(self.show_topology)
        button_layout.addWidget(self.btn_topology)

        self.btn_grid = QPushButton("show spectrum grid")
        self.btn_grid.pressed.connect(self.show_grid)
        button_layout.addWidget(self.btn_grid)

        # self.button.clicked.connect(self.show_new_window)
        # self.setCentralWidget(self.button)
        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    def show_grid(self):
        self.plot_grid = PlotGrid(self)
        self.plot_grid.show()

    def show_topology(self):
        self.plot_topology = PlotTopology(self)
        self.plot_topology.show()

    def advance_arrivals(self):
        self.btn_advance.setEnabled(False)
        self.btn_advance.setText("running....")
        oldest_scenario = OldestFirst(10, 10)
        evaluate_heuristic(self.env, oldest_scenario.choose_oldest_first, n_eval_episodes=2)
        self.btn_advance.setEnabled(True)
        self.btn_advance.setText("advance")

    def start_drl(self):
        pass

    def reset_env(self):
        pass

    def step(self):
        action = self.agent.predict(env.observation())
        self.env.step(action)


class PlotTopology(QWidget):
    def __init__(self, parent_window):
        super().__init__()

        self.parent_window = parent_window
        self.setWindowTitle("topology plot")
        # self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.figure = plt.figure()
        self.sc = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.sc)
        # Plot the NetworkX graph on the Matplotlib canvas
        G = self.parent_window.env.env.env.topology
        pos = nx.spring_layout(G)  # You can choose a layout algorithm here
        nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1000)
        self.sc.draw()


class PlotGrid(QWidget):
    def __init__(self, parent_window):
        super().__init__()

        self.parent_window = parent_window
        self.setWindowTitle("Spectrum assignment plot")
        # self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.figure = plt.figure(figsize=(15, 10))
        self.sc = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.sc)
        topology = self.parent_window.env.env.env.topology
        slot_allocation = self.parent_window.env.env.env.spectrum_slots_allocation
        # Plot the spectrum assignment graph
        plot_spectrum_assignment_on_canvas(topology, slot_allocation, self.sc, values=True, title="Spectrum Assignment")


def plot_spectrum_assignment_on_canvas(topology, vector, canvas, values=False, title=None):
    # Create a Matplotlib figure and use the provided canvas
    fig = canvas.figure
    fig.clf()
    ax = fig.add_subplot(111)

    cmap = copy.copy(plt.cm.viridis)
    cmap.set_under(color='white')

    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')
    p = ax.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')

    if values:
        thresh = vector.max() / 2.
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] == -1:
                continue
            else:
                text = '{:.0f}'.format(vector[i, j])
            color = cmap_reverse(vector[i, j] / vector.max())
            diff_color = np.sum(np.array(color) - np.array(cmap(vector[i, j] / vector.max())))
            if np.abs(diff_color) < 0.1:
                red = max(color[0] + 0.5, 1.)
                green = max(color[1] - 0.5, 0.)
                blue = max(color[2] - 0.5, 0.)
                color = (red, blue, green)
            #             print(i, j, vector[i, j], diff_color)
            plt.text(j + 0.5, i + 0.5, text,
                     horizontalalignment="center", verticalalignment='center',
                     color=color)

    ax.set_xlabel('Frequency slot')
    ax.set_ylabel('Link')
    if title is not None:
        plt.title(title)
    #     plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [link[0] + '-' + link[1] for link in topology.edges()])
    plt.yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
               [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])
    #     plt.colorbar()
    plt.tight_layout()
    plt.xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])

    canvas.draw()
