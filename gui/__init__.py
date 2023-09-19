import copy
import itertools
import pickle
from random import randint

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtGui import QFont

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import matplotlib.colors as mcolors

from optical_rl_gym.utils import evaluate_heuristic
from optical_rl_gym.envs.deepdefragmentation_env import populate_network
from optical_rl_gym.envs.defragmentation_env import OldestFirst

from gui.workers import Worker, RunArrivalsWorker


def run_arrivals(env):
    # oldest_scenario = OldestFirst(10, 10)
    # evaluate_heuristic(env, oldest_scenario.choose_oldest_first, n_eval_episodes=2)
    evaluate_heuristic(env, populate_network, n_eval_episodes=1)
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

        self.topology_plot = self.main_window.plot_topology(True)
        self.topology_plot.setFixedSize(400, 500)
        spectrum_layout.addWidget(self.topology_plot)
        self.grid_plot = self.main_window.plot_grid()
        self.grid_plot.setFixedSize(1400, 500)
        spectrum_layout.addWidget(self.grid_plot)

        label_text = f"demand ID is {self.main_window.env.env.env.previous_service.service_id} \n from source {self.main_window.env.env.env.previous_service.source} to destination {self.main_window.env.env.env.previous_service.destination}. \n The route" \
                     f"is {self.main_window.env.env.env.previous_service.route.node_list} and \n the initial slot is {self.main_window.env.env.env.previous_service.initial_slot}"
        label = QLabel(label_text)
        font = QFont()
        font.setPointSize(16)  # Set the desired font size
        label.setFont(font)
        label.setFixedSize(400, 500)
        spectrum_layout_no.addWidget(label)

        self.grid_plot_no = self.main_window.plot_grid(False)
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
        self.blocked = env.env.env.blocked_services[-400:]
        self.blocked_nodefrag = env.env.env.blocked_services[-400:]
        self.r_frag_nodefrag = env.env.env.rfrag_after_list[-400:]
        self.r_frag = env.env.env.rfrag_after_list[-400:]
        self.reward = env.env.env.rewards[-400:]
        self.reward_nodefrag = env.env.env.rewards[-400:]
        self.env = env
        self.env_no_df = copy.deepcopy(env)
        self.agent = agent
        self.tapi_client = tapi_client
        self.w = None  # No external window yet.
        self.continue_flag = False  # to Continue the DRL-based defragmentation.
        self.worker_cnt = None

        # temporarily, we start the env from zero, so we need to run it for a while to plot spectrum grid. later on,
        # we will start from a saved environment
        # evaluate_heuristic(self.env, populate_network, n_eval_episodes=80)
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
        self.topology_plot.setFixedSize(400, 500)
        topology_layout.addWidget(self.topology_plot)
        self.grid_plot = self.plot_grid()
        self.grid_plot.setFixedSize(1400, 500)
        topology_layout.addWidget(self.grid_plot)

        self.btn_advance = QPushButton("advance ")
        self.btn_advance.pressed.connect(self.advance_arrivals)
        button_layout.addWidget(self.btn_advance)

        self.btn_drl = QPushButton("start DRL-based defragmentation")
        self.btn_drl.pressed.connect(self.start_drl)
        button_layout.addWidget(self.btn_drl)

        self.btn_drl_cnt = QPushButton("Continue DRL-based defragmentation")
        self.btn_drl_cnt.pressed.connect(self.continue_drl)
        button_layout.addWidget(self.btn_drl_cnt)

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
        # self.shanon_plot = self.plot_shanon()
        # metric_layout.addWidget(self.shanon_plot)
        #
        self.blocked_plot = self.plot_blocked()
        metric_layout.addWidget(self.blocked_plot)

        self.reward_plot = self.plot_reward()
        metric_layout.addWidget(self.reward_plot)

        pagelayout.addLayout(topology_layout)
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(metric_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        ### to show it in a full screen mode!

        self.showFullScreen()

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
        self.env_no_df = copy.deepcopy(env)
        self.btn_advance.setEnabled(True)
        self.btn_advance.setText("advance")

    def start_drl(self):

        worker = RunArrivalsWorker(
            self.env,
            self.env_no_df,
            self.agent
        )

        worker.signals.result.connect(self.update_plots)
        self.threadpool.start(worker)

    def continue_drl(self):

        self.worker_cnt = RunArrivalsWorker(
            self.env,
            self.env_no_df,
            self.agent,
            False
        )

        self.worker_cnt.signals.result.connect(self.update_plots)
        self.threadpool.start(self.worker_cnt)


    def start_drl(self):

        worker = RunArrivalsWorker(
            self.env,
            self.env_no_df,
            self.agent
        )

        worker.signals.result.connect(self.update_plots)
        self.threadpool.start(worker)

    def update_plots(self, result):
        self.x_data += 10
        self.env = result[0]
        self.env_no_df = result[1]
        flag_blocking = result[2]
        r_frag_update = result[3]
        r_frag_nodefrag_update = result[4]
        reward_update = result[5]
        reward_nodefrag_update = result[6]

        blocked_update = result[7]
        blocked_nodefrag_update = result[8]
        a = result[9]
        b = result[10]
        c = result[11]
        d = result[12]

        if flag_blocking is False or self.continue_flag:
            self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation)
            self.update_rfrag(r_frag_update, r_frag_nodefrag_update)
            self.update_reward(reward_update, reward_nodefrag_update)
            self.update_blocked(blocked_update, blocked_nodefrag_update, b, c)
        else:
            self.continue_flag = True
            self.another_window = AnotherWindow(self)
            self.another_window.show()

    def reset_env(self):
        pass
        # if self.worker_cnt:
        #     self.worker_cnt.stop()
        #     self.threadpool.waitForDone()

    def plot_topology(self, highlight=False):
        figure = plt.figure()
        sc = FigureCanvasQTAgg(figure)
        # Plot the NetworkX graph on the Matplotlib canvas
        G = self.env.env.env.topology
        pos = nx.spring_layout(G, seed=40)  # You can choose a layout algorithm here
        nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1000)

        if highlight:
            nodes = self.env.env.env.previous_service.route.node_list
            for i in range(len(nodes) - 1):
                nx.draw_networkx_edges(G, pos, edgelist=[(nodes[i], nodes[i + 1])], edge_color='red', width=2)

        return sc

    def plot_grid(self, drl=True):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        topology = self.env.env.env.topology
        if drl:
            slot_allocation = self.env.env.env.spectrum_slots_allocation
            title = "Spectrum Assignment"
        else:
            slot_allocation = self.env_no_df.env.env.spectrum_slots_allocation
            title = "Spectrum Assignment for No defragmentation scenario"
        # Plot the spectrum assignment graph
        return plot_spectrum_assignment_on_canvas(topology, slot_allocation, sc, values=True,
                                                  title=title)

    def plot_rfrag(self):
        self.x_data = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(self.x_data, self.r_frag, label='RSS metric DRL', color='blue')
        ax.plot(self.x_data, self.r_frag_nodefrag, label='RSS metric No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("RSS metric")
        ax.set_title("RSS metric")
        ax.legend()
        sc.draw()
        return sc

    def update_rfrag(self, r_frag_update, r_frag_nodefrag_update):
        self.rfrag_plot.figure.clf()
        ax = self.rfrag_plot.figure.gca()
        r_frag_update_array = np.array(r_frag_update)
        temporarily_array = self.r_frag[len(r_frag_update_array):]
        self.r_frag = np.concatenate((temporarily_array, r_frag_update_array))
        r_frag_update_array_nodefrag = np.array(r_frag_nodefrag_update)
        temporarily_nodefrag_array = self.r_frag[len(r_frag_update_array_nodefrag):]
        self.r_frag_nodefrag = np.concatenate((temporarily_nodefrag_array, r_frag_update_array_nodefrag))
        ax.plot(self.x_data, self.r_frag, label='RSS metric DRL', color='blue')
        ax.plot(self.x_data, self.r_frag_nodefrag, label='RSS metric No Defrag', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("RSS metric")
        ax.set_title("RSS metric")
        ax.legend()
        self.rfrag_plot.draw()

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

    def plot_blocked(self):
        x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        ax.plot(x, self.blocked, label='Sum of blocked services DRL', color='blue')
        ax.plot(x, self.blocked, label='Sum of blocked services No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        sc.draw()
        return sc

    def update_blocked(self, blocked_update, blocked_nodefrag_update, b, c):
        self.blocked_plot.figure.clf()
        ax = self.blocked_plot.figure.gca()


        # blocked_update_array = np.array(blocked_update)
        # blocked_update_array += self.blocked[len(self.blocked)-1]

        temporarily_array = self.blocked[len(blocked_update):]
        # self.blocked = np.concatenate((temporarily_array, blocked_update_array))
        temporarily_array.extend(blocked_update)
        self.blocked = temporarily_array
        # blocked_update_array_nodefrag = np.array(blocked_nodefrag_update)

        # blocked_update_array_nodefrag += self.blocked_nodefrag[len(self.blocked_nodefrag)-1]
        temporarily_nodefrag_array = self.blocked_nodefrag[len(blocked_nodefrag_update):]
        temporarily_nodefrag_array.extend(blocked_nodefrag_update)
        # self.blocked_nodefrag = np.concatenate((temporarily_nodefrag_array, blocked_update_array_nodefrag))
        self.blocked_nodefrag = temporarily_nodefrag_array




        ax.plot(self.x_data, self.blocked, label='Number of blocked services DRL', color='blue')
        ax.plot(self.x_data, self.blocked_nodefrag, label='Number of blocked services No Defrag', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("blocked services")
        ax.set_title("blocked services")
        ax.legend()
        self.blocked_plot.draw()

    def plot_reward(self):
        x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        sum_rewards = []
        cumulative_sum = 0
        for i, value in enumerate(self.reward):
            cumulative_sum += value
            sum_rewards.append(cumulative_sum)

        ax.plot(x, sum_rewards, label='Sum of rewards for DRL', color='blue')
        ax.plot(x, sum_rewards, label='Sum of rewards for No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        sc.draw()
        return sc

    def update_reward(self, reward_update, reward_nodefrag_update):
        self.reward_plot.figure.clf()
        ax = self.reward_plot.figure.gca()
        reward_update_array = np.array(reward_update)
        temporarily_array = self.reward[len(reward_update_array):]
        self.reward = np.concatenate((temporarily_array, reward_update_array))
        reward_update_array_nodefrag = np.array(reward_nodefrag_update)
        temporarily_nodefrag_array = self.reward_nodefrag[len(reward_update_array_nodefrag):]
        self.reward_nodefrag = np.concatenate((temporarily_nodefrag_array, reward_update_array_nodefrag))

        sum_rewards = []
        cumulative_sum = 0
        for i, value in enumerate(self.reward):
            cumulative_sum += value
            sum_rewards.append(cumulative_sum)
        sum_rewards_nodefrag = []
        cumulative_sum_nodefrag = 0
        for i, value in enumerate(self.reward_nodefrag):
            cumulative_sum_nodefrag += value
            sum_rewards_nodefrag.append(cumulative_sum)

        if len(sum_rewards) > 400:
            a = 1
        ax.plot(self.x_data, sum_rewards, label='Sum of rewards for DRL', color='blue')
        ax.plot(self.x_data, sum_rewards_nodefrag, label='Sum of rewards No Defrag', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        self.reward_plot.draw()


    def update_grid(self, topology, slot_allocation):
        canvas = self.grid_plot
        canvas.figure.clf()  # Clear the previous plot

        cmap = plt.cm.get_cmap("tab20")
        cmap.set_under(color='white')
        cmap_reverse = plt.cm.viridis_r
        cmap_reverse.set_under(color='black')
        # https://stackoverflow.com/questions/7164397/find-the-min-max-excluding-zeros-in-a-numpy-array-or-a-tuple-in-python
        masked_a = np.ma.masked_equal(slot_allocation, -1, copy=False)
        norm = mcolors.LogNorm(vmin=masked_a.min(), vmax=slot_allocation.max())

        # p = ax.pcolor(vector, cmap=cmap, norm=norm, edgecolors='gray')
        p = canvas.figure.add_subplot(111).pcolor(slot_allocation, cmap=cmap, norm=norm, edgecolors='gray')

        ax = canvas.figure.gca()
        ax.set_xlabel('Frequency slot')
        ax.set_ylabel('Link')
        plt.yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
                   [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])

        plt.title("Spectrum Assignment")  # Set the title
        plt.xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
        plt.tight_layout()
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

    # p = ax.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')
    masked_a = np.ma.masked_equal(vector, -1, copy=False)
    norm = mcolors.LogNorm(vmin=masked_a.min(), vmax=vector.max())

    p = ax.pcolor(vector, cmap=cmap, norm=norm, edgecolors='gray')

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
        ax.set_title(title)
    #     plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [link[0] + '-' + link[1] for link in topology.edges()])
    ax.set_yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
               [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])
    #     plt.colorbar()
    ax.set_xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
    plt.tight_layout()
    # canvas.draw()
    return canvas
