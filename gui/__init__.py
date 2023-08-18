import traceback

from multiprocessing import Process

from PyQt6.QtCore import pyqtSignal, QRunnable, pyqtSlot, QThreadPool, QObject
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout

from defrag.tests.test_defrag_dashboard import test_dashboard
import sys
import networkx as nx
import matplotlib.pyplot as plt
from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.envs.defragmentation_env import OldestFirst
from optical_rl_gym.envs.deepdefragmentation_env import populate_network
import copy
import numpy as np
import itertools

from random import randint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib.figure import Figure


def run_arrivals(env):
    # oldest_scenario = OldestFirst(10, 10)
    # evaluate_heuristic(env, oldest_scenario.choose_oldest_first, n_eval_episodes=2)
    evaluate_heuristic(env, populate_network, n_eval_episodes=2)
    return env


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


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
        self.env_no_df = env
        self.agent = agent
        self.tapi_client = tapi_client
        self.w = None  # No external window yet.

        # temporarily, we start the env from zero, so we need to run it for a while to plot spectrum grid. later on,
        # we will start from a saved environment
        evaluate_heuristic(self.env, populate_network, n_eval_episodes=2)
        self.threadpool = QThreadPool()

        self.setWindowTitle("DefragDashboard")
        pagelayout = QVBoxLayout()
        topology_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        # First horizontal layout
        topology_plot = self.plot_topology()
        topology_layout.addWidget(topology_plot)
        grid_plot = self.plot_grid()
        topology_layout.addWidget(grid_plot)

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

        pagelayout.addLayout(topology_layout)
        pagelayout.addLayout(button_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    def show_grid(self):
        pass

    def show_topology(self):
        pass

    def advance_arrivals(self):
        worker = Worker(run_arrivals, self.env)
        worker.signals.result.connect(self.stop_arrivals)
        self.threadpool.start(worker)
        # worker.finished.connect(self.stop_arrivals)
        self.btn_advance.setEnabled(False)
        self.btn_advance.setText("running....")

    def stop_arrivals(self, env):
        self.env = env
        self.btn_advance.setEnabled(True)
        self.btn_advance.setText("advance")

    def start_drl(self):
        for _ in range(5):
            obs_drl = self.env.reset()
            obs_no_df = self.env_no_df.reset()
            done, state = False, None
            while not done:
                action, _states = self.agent.predict(obs_drl, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                if reward == -1:
                    obs, reward, done, info = self.env.step(0)

    def reset_env(self):
        pass

    def plot_topology(self):
        figure = plt.figure()
        sc = FigureCanvasQTAgg(figure)
        # Plot the NetworkX graph on the Matplotlib canvas
        G = self.env.env.env.topology
        pos = nx.spring_layout(G)  # You can choose a layout algorithm here
        nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1000)
        return sc

    def plot_grid(self):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        topology = self.env.env.env.topology
        slot_allocation = self.env.env.env.spectrum_slots_allocation
        # Plot the spectrum assignment graph
        return plot_spectrum_assignment_on_canvas(topology, slot_allocation, sc, values=True, title="Spectrum Assignment")


    # def step(self):
    #     action = self.agent.predict(env.observation())
    #     self.env.step(action)



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

    # canvas.draw()
    return canvas
