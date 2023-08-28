import copy
import itertools
import pickle
from random import randint

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.envs.deepdefragmentation_env import populate_network
from optical_rl_gym.envs.defragmentation_env import OldestFirst

from gui.workers import Worker, RunArrivalsWorker

def run_arrivals(env):
    # oldest_scenario = OldestFirst(10, 10)
    # evaluate_heuristic(env, oldest_scenario.choose_oldest_first, n_eval_episodes=2)
    # evaluate_heuristic(env, populate_network, n_eval_episodes=2)
    return env


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.layout = QVBoxLayout()
        spectrum_layout = QHBoxLayout()
        spectrum_layout_no = QHBoxLayout()

        self.topology_plot = self.main_window.plot_topology()
        spectrum_layout.addWidget(self.topology_plot)
        self.grid_plot = self.main_window.plot_grid()
        spectrum_layout.addWidget(self.grid_plot)

        self.grid_plot_no = self.main_window.plot_grid()
        spectrum_layout_no.addWidget(self.grid_plot_no)


        self.layout.addLayout(spectrum_layout)
        self.layout.addLayout(spectrum_layout_no)

        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    """
    Responsibilities:
    - call functions
    - all plotting
    """

    def __init__(self, env, agent, tapi_client):
        super().__init__()
        self.env = env
        self.env_no_df = copy.deepcopy(env)
        self.agent = agent
        self.tapi_client = tapi_client
        self.w = None  # No external window yet.

        # temporarily, we start the env from zero, so we need to run it for a while to plot spectrum grid. later on,
        # we will start from a saved environment
        # evaluate_heuristic(self.env, populate_network, n_eval_episodes=30)
        # with open('fragmented_env.pickle', 'wb') as file:
        #     pickle.dump(env, file)


        self.threadpool = QThreadPool()

        self.setWindowTitle("DefragDashboard")
        pagelayout = QVBoxLayout()
        topology_layout = QHBoxLayout()
        button_layout = QHBoxLayout()
        metric_layout = QHBoxLayout()

        # First horizontal layout
        self.canvas_grid = None
        self.topology_plot = self.plot_topology()
        topology_layout.addWidget(self.topology_plot)
        self.grid_plot = self.plot_grid()
        topology_layout.addWidget(self.grid_plot)

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

        self.rfrag_plot = self.plot_rfrag()
        metric_layout.addWidget(self.rfrag_plot)
        self.shanon_plot = self.plot_shanon()
        metric_layout.addWidget(self.shanon_plot)
        self.noc_plot = self.plot_noc()
        metric_layout.addWidget(self.noc_plot)

        pagelayout.addLayout(topology_layout)
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(metric_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    def show_grid(self):
        pass

    def show_topology(self):
        pass

    # TODO: create a method that uses RunArrivalsWorker

    def advance_arrivals(self):
        worker = Worker(
            run_arrivals,  # function to be executed within the process
            self.env  # args to be sent to the function
        )
        worker.signals.result.connect(self.stop_arrivals)  # function to be called when the process yields a result
        self.threadpool.start(worker)
        # worker.finished.connect(self.stop_arrivals)
        self.btn_advance.setEnabled(False)
        self.btn_advance.setText("running....")

    def stop_arrivals(self, env):
        self.env = env
        self.btn_advance.setEnabled(True)
        self.btn_advance.setText("advance")

    def start_drl(self):
        # env_no_df = self.env_no_df
        a = 0
        b = 0
        c = 0
        d = 0
        for _ in range(1):
            # if a > 0 :
            #     break
            obs_drl = self.env.reset()
            obs_no_df = self.env_no_df.reset()
            done, state = False, None
            while not done:
                action, _states = self.agent.predict(obs_drl, deterministic=True)
                obs_drl, reward, done, info = self.env.step(action)
                if action == 0:
                    obs_no_df, reward_df, done_df, info_df = self.env_no_df.step(0)

                    if self.env_no_df.env.env.previous_service_accepted == False and self.env.env.env.previous_service_accepted == True:
                        a += 1
                        # self.another_window = AnotherWindow(self)
                        # self.another_window.show()
                        # break

                    # if self.env_no_df.env.env.previous_service_accepted == True and self.env.env.env.previous_service_accepted == False:
                    #     b += 1
                    # if self.env_no_df.env.env.previous_service_accepted == True and self.env.env.env.previous_service_accepted == True:
                    #     c += 1
                    #
                    # if self.env_no_df.env.env.previous_service_accepted == False and self.env.env.env.previous_service_accepted == False:
                    #     d +=1


                self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation )
                if reward == -1:
                    obs_drl, reward, done_2, info = self.env.step(0)
                    obs_no_df, reward_df, done_df, info_df = self.env_no_df.step(0)

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

    def plot_rfrag(self):
        x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(x, self.env.env.env.rfrag_after_list, label='RSS metric')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("RSS metric")
        ax.set_title("RSS metric")
        ax.legend()
        sc.draw()
        return sc

    def plot_shanon(self):
        x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(x, self.env.env.env.shanon_entrophy_after_list, label='Shanon metric')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Shanon metric")
        ax.set_title("Shanon metric")
        ax.legend()
        sc.draw()
        return sc

    def plot_noc(self):
        x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(x, self.env.env.env.num_cut_list_after, label='NoC metric')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("NoC metric")
        ax.set_title("NoC metric")
        ax.legend()
        sc.draw()
        return sc

    # def update_grid(self):
        # topology = self.env.env.env.topology
        # slot_allocation = self.env.env.env.spectrum_slots_allocation
        # Plot the spectrum assignment graph
        # canvas = plot_spectrum_assignment_on_canvas(topology, slot_allocation, sc, values=True, title="Spectrum Assignment")
        # if self.grid_plot:
        #     self.grid_plot.close()
        #     # self.grid_plot = None
        #     main_layout = self.centralWidget().layout()
        #     main_layout.removeWidget(self.grid_plot)
        #
        # self.grid_plot = self.plot_grid()
        # main_layout = self.centralWidget().layout()
        # main_layout.addWidget(self.grid_plot)

    def update_grid(self, topology, slot_allocation):
        canvas = self.grid_plot
        canvas.figure.clf()  # Clear the previous plot

        cmap = plt.cm.get_cmap("tab20")
        cmap.set_under(color='white')
        cmap_reverse = plt.cm.viridis_r
        cmap_reverse.set_under(color='black')
        p = canvas.figure.add_subplot(111).pcolor(slot_allocation, cmap=cmap, vmin=-0.0001, edgecolors='gray')

        ax = canvas.figure.gca()
        ax.set_xlabel('Frequency slot')
        ax.set_ylabel('Link')
        plt.yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
                   [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])

        plt.title("Spectrum Assignment")  # Set the title
        plt.tight_layout()
        plt.xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])

        canvas.draw()  # Redraw the canvas

        return canvas


    # def step(self):
    #     action = self.agent.predict(env.observation())
    #     self.env.step(action)



def plot_spectrum_assignment_on_canvas(topology, vector, canvas, values=False, title=None):
    # Create a Matplotlib figure and use the provided canvas
    fig = canvas.figure
    fig.clf()
    ax = fig.add_subplot(111)

    cmap = copy.copy(plt.cm.viridis)
    # cmap = plt.cm.get_cmap("tab20")
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
            # plt.text(j + 0.5, i + 0.5, text,
            #          horizontalalignment="center", verticalalignment='center',
            #          color=color)

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
