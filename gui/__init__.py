import copy
import itertools
import pickle
from random import randint
import json
import time
import re
import pprint

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QTextEdit, QDialog
from PyQt6.QtGui import QFont
import requests

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


class TapiWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, main_window, json_data, json_data_delete):
        super().__init__()

        self.main_window = main_window

        self.layout = QVBoxLayout()
        create_message = QHBoxLayout()
        delete_message = QHBoxLayout()

        nodes = self.main_window.env.env.env.last_service_to_defrag.route.node_list

        self.topology_plot = self.main_window.plot_topology(True, nodes=nodes)
        self.topology_plot.setFixedSize(400, 500)
        create_message.addWidget(self.topology_plot)

        panel = QWidget()
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)
        title_label = QLabel("Create a connectivity service TAPI message")
        font = QFont()
        font.setPointSize(16)  # Set the desired font size for the title
        title_label.setFont(font)

        # create_message.addWidget(title_label)
        panel_layout.addWidget(title_label)

        sd = [self.main_window.env.env.env.last_service_to_defrag.source,
              self.main_window.env.env.env.last_service_to_defrag.destination]
        central_freq = self.main_window.env.env.env.last_service_to_defrag.initial_slot + self.main_window.env.env.env.last_service_to_defrag.number_slots
        for i, endpoint in enumerate(json_data["tapi-connectivity:connectivity-service"][0]["end-point"]):
            endpoint["service-interface-point"]["service-interface-point-uuid"] = sd[i]
            endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"]["channel"][
                "central-frequency"] = f"{central_freq}"
            endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"][
                "rx-channel"]["central-frequency"] = f"{central_freq}"

        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(10, 10, 580, 380)
        # self.text_edit.setPlainText(json.dumps(json_data, indent=4))
        text_create = "http POST 'https://172.20.132.200:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context'\n"
        text_create += json.dumps(json_data, indent=4)
        # self.text_edit.setPlainText(json.dumps(json_data, indent=4))
        self.text_edit.setPlainText(text_create)
        panel_layout.addWidget(self.text_edit)
        create_message.addWidget(panel)


        route = " -> ".join(self.main_window.env.env.env.last_service_to_defrag.route.node_list)
        label_text = f"demand ID is {self.main_window.env.env.env.last_service_to_defrag.service_id} \n from source {self.main_window.env.env.env.last_service_to_defrag.source} to destination {self.main_window.env.env.env.last_service_to_defrag.destination}. \n The route" \
                     f" is {route}, \n the old initial slot is {self.main_window.env.env.env.last_old_initial_slot} and \n the new initial slot is {self.main_window.env.env.env.last_new_initial_slot}"

        label = QLabel(label_text)
        font = QFont()
        font.setPointSize(16)  # Set the desired font size
        label.setFont(font)
        label.setFixedSize(400, 500)
        delete_message.addWidget(label)

        panel_delete = QWidget()
        panel_layout_delete = QVBoxLayout()
        panel_delete.setLayout(panel_layout_delete)


        title_label = QLabel("Delete a connectivity service TAPI message")
        font = QFont()
        font.setPointSize(16)  # Set the desired font size for the title
        title_label.setFont(font)

        # delete_message.addWidget(title_label)
        panel_layout_delete.addWidget(title_label)
        json_data_delete["tapi-connectivity:input"][
            "tapi-connectivity:service-id-or-name"] = self.main_window.env.env.env.last_service_to_defrag.service_id

        delete_url = "http DELETE 'https://172.20.132.200:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/tapi-connectivity:connectivity-service=00000070-0000-0000-0000-000000146732"
        pattern = r'(connectivity-service=.+)'
        replacement = f'connectivity-service={self.main_window.env.env.env.last_service_to_defrag.service_id}"'
        result_text = re.sub(pattern, replacement, delete_url)

        self.text_edit_delete = QTextEdit(self)
        # self.text_edit_delete.setPlainText(json.dumps(json_data_delete, indent=4))
        self.text_edit_delete.setPlainText(result_text)
        panel_layout_delete.addWidget(self.text_edit_delete)
        delete_message.addWidget(panel_delete)

        self.layout.addLayout(create_message)
        self.layout.addLayout(delete_message)

        self.setLayout(self.layout)


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
        nodes = self.main_window.env.env.env.previous_service.route.node_list
        self.topology_plot = self.main_window.plot_topology(True, nodes)
        self.topology_plot.setFixedSize(400, 500)
        spectrum_layout.addWidget(self.topology_plot)
        self.grid_plot = self.main_window.plot_grid()
        self.grid_plot.setFixedSize(1400, 500)
        spectrum_layout.addWidget(self.grid_plot)

        route = " -> ".join(self.main_window.env.env.env.previous_service.route.node_list)

        label_text = f"demand ID is {self.main_window.env.env.env.previous_service.service_id} \n from source {self.main_window.env.env.env.previous_service.source} to destination {self.main_window.env.env.env.previous_service.destination}. \n The route" \
                     f" is {route}, \n the initial slot is {self.main_window.env.env.env.previous_service.initial_slot}"
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


class FirstAllocationWindow(QWidget):
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
        nodes = self.main_window.env.env.env.last_service_to_defrag.route.node_list
        self.topology_plot = self.main_window.plot_topology(True, nodes)
        self.topology_plot.setFixedSize(400, 500)
        spectrum_layout.addWidget(self.topology_plot)
        self.grid_plot_before = self.main_window.plot_grid(True, True)
        self.grid_plot_before.setFixedSize(1400, 500)
        spectrum_layout.addWidget(self.grid_plot_before)

        route = " -> ".join(self.main_window.env.env.env.last_service_to_defrag.route.node_list)

        label_text = f"demand ID is {self.main_window.env.env.env.last_service_to_defrag.service_id} \n from source {self.main_window.env.env.env.last_service_to_defrag.source} to destination {self.main_window.env.env.env.last_service_to_defrag.destination}. \n The route" \
                     f" is {route}, \n the old initial slot is {self.main_window.env.env.env.last_old_initial_slot} and \n the new initial slot is {self.main_window.env.env.env.last_new_initial_slot}"
        label = QLabel(label_text)
        font = QFont()
        font.setPointSize(16)  # Set the desired font size
        label.setFont(font)
        label.setFixedSize(400, 500)
        spectrum_layout_no.addWidget(label)

        self.grid_plot_after = self.main_window.plot_grid(True, False)
        spectrum_layout_no.addWidget(self.grid_plot_after)

        self.layout.addLayout(spectrum_layout)
        self.layout.addLayout(spectrum_layout_no)

        self.setLayout(self.layout)


class DetailWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.setWindowTitle("Detailed figures")
        self.layout = QVBoxLayout()
        first_layout = QHBoxLayout()
        second_layout = QHBoxLayout()

        self.rfrag_plot = self.main_window.plot_figure(self.main_window.r_frag, self.main_window.r_frag_nodefrag,
                                                       'RSS metric')
        first_layout.addWidget(self.rfrag_plot)

        self.shanon_plot = self.main_window.plot_figure(self.main_window.env.env.env.shanon_entrophy_after_list,
                                                        self.main_window.env_no_df.env.env.shanon_entrophy_after_list,
                                                        'Shannon entropy metric')
        first_layout.addWidget(self.shanon_plot)

        # self.noc_plot = self.main_window.plot_noc()
        # first_layout.addWidget(self.shanon_plot)

        self.reward_plot = self.main_window.plot_reward()
        first_layout.addWidget(self.reward_plot)
        #
        self.blocked_plot = self.main_window.plot_figure(self.main_window.blocked, self.main_window.blocked_nodefrag,
                                                         'Number of blocked services')
        second_layout.addWidget(self.blocked_plot)

        self.moves_plot = self.main_window.plot_figure(self.main_window.moves, self.main_window.moves_nodefrag,
                                                       'Number of connection reallocations')
        second_layout.addWidget(self.moves_plot)

        self.cycles_plot = self.main_window.plot_figure(self.main_window.env.env.env.defragmentation_procedure_list,
                                                        self.main_window.env_no_df.env.env.defragmentation_procedure_list,
                                                        'Number of defragmentation cycles')
        second_layout.addWidget(self.cycles_plot)

        self.layout.addLayout(first_layout)
        self.layout.addLayout(second_layout)

        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    """
    Responsibilities:
    - call functions
    - all plotting
    """

    def __init__(self, env, agent, tapi_client):
        super().__init__()
        self.blocked = copy.deepcopy(env.env.env.blocked_services)
        self.blocked_nodefrag = copy.deepcopy(env.env.env.blocked_services)
        self.r_frag_nodefrag = copy.deepcopy(env.env.env.rfrag_after_list)
        self.r_frag = copy.deepcopy(env.env.env.rfrag_after_list)
        self.reward = copy.deepcopy(env.env.env.rewards)
        self.reward_nodefrag = copy.deepcopy(env.env.env.rewards)
        self.moves = copy.deepcopy(env.env.env.num_moves_list)
        self.moves_nodefrag = copy.deepcopy(env.env.env.num_moves_list)
        # self.cycles = env.env.env.defragmentation_procedure_list[-400:]
        # self.cycles_nodefrag = env.env.env.defragmentation_procedure_list[-400:]
        self.env = copy.deepcopy(env)
        self.env_no_df = copy.deepcopy(env)
        self.saved_env = copy.deepcopy(env)
        self.agent = agent
        self.tapi_client = tapi_client
        self.continue_flag = False  # to Continue the DRL-based defragmentation.
        self.worker_cnt = None
        self.x_data = np.arange(-400, 0)
        self.slot_allocation_before_action = None

        self.drl_stop = True  ## I define this to control behaviour of continue defragmentation

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

        self.btn_advance = QPushButton("Advance arrivals ")
        self.btn_advance.pressed.connect(self.advance_arrivals)
        button_layout.addWidget(self.btn_advance)

        self.btn_drl = QPushButton("Start DRL-based defragmentation")
        self.btn_drl.pressed.connect(self.start_drl)
        button_layout.addWidget(self.btn_drl)

        self.btn_drl_cnt = QPushButton("Continue DRL-based defragmentation")
        self.btn_drl_cnt.pressed.connect(self.continue_drl)
        button_layout.addWidget(self.btn_drl_cnt)

        self.btn_stop = QPushButton("Stop DRL-based defragmentation")
        self.btn_stop.pressed.connect(self.stop_drl)
        button_layout.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.pressed.connect(self.reset_env)
        button_layout.addWidget(self.btn_reset)

        self.btn_tapi = QPushButton("Show TAPI message")
        self.btn_tapi.pressed.connect(self.show_tapi)
        button_layout.addWidget(self.btn_tapi)

        self.btn_detail = QPushButton("Show details")
        self.btn_detail.pressed.connect(self.show_detail)
        button_layout.addWidget(self.btn_detail)

        self.rfrag_plot = self.plot_figure(self.r_frag, self.r_frag_nodefrag, 'RSS metric')
        metric_layout.addWidget(self.rfrag_plot)

        #
        self.blocked_plot = self.plot_figure(self.blocked, self.blocked_nodefrag, 'Number of blocked services')
        metric_layout.addWidget(self.blocked_plot)

        # self.reward_plot = self.plot_reward()
        # metric_layout.addWidget(self.reward_plot)
        self.moves_plot = self.plot_figure(self.moves, self.moves_nodefrag, 'Number of connection reallocations')
        metric_layout.addWidget(self.moves_plot)

        pagelayout.addLayout(topology_layout)
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(metric_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)
        ### to show it in a full screen mode!

        self.showFullScreen()

    def stop_drl(self):
        if self.worker_cnt:
            self.worker_cnt.stop()

    def show_tapi(self):

        path_create = "tapi/tapi-create.json"
        path_delete = "tapi/tapi-delete.json"
        json_data = load_json_from_file(path_create)
        json_data_delete = load_json_from_file(path_delete)
        self.tapi_window = TapiWindow(self, json_data, json_data_delete)
        self.tapi_window.show()

    def show_detail(self):

        self.tapi_window = DetailWindow(self)
        self.tapi_window.show()

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
            self.drl_stop,
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
        moves_update = result[9]
        moves_update_no_defrag = result[10]
        self.drl_stop = result[11]
        self.slot_allocation_before_action = result[12]
        flag_first_allocation = result[13]

        if flag_first_allocation:
            self.another_window = FirstAllocationWindow(self)
            self.another_window.show()

        else:

            if flag_blocking is False or self.continue_flag:
                self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation)
                self.update_rfrag(r_frag_update, r_frag_nodefrag_update)
                # self.update_reward(reward_update, reward_nodefrag_update)
                self.update_moves(moves_update, moves_update_no_defrag)
                self.update_blocked(blocked_update, blocked_nodefrag_update)
            else:
                self.continue_flag = True
                self.another_window = AnotherWindow(self)
                self.another_window.show()

    def reset_env(self):
        self.env = copy.deepcopy(self.saved_env)
        self.env_no_df = copy.deepcopy(self.saved_env)

        self.blocked = copy.deepcopy(self.env.env.env.blocked_services)
        self.blocked_nodefrag = copy.deepcopy(self.env.env.env.blocked_services)
        self.r_frag_nodefrag = copy.deepcopy(self.env.env.env.rfrag_after_list)
        self.r_frag = copy.deepcopy(self.env.env.env.rfrag_after_list)
        self.reward = copy.deepcopy(self.env.env.env.rewards)
        self.reward_nodefrag = copy.deepcopy(self.env.env.env.rewards)
        self.drl_stop = True
        self.continue_flag = False  # to Continue the DRL-based defragmentation. It should be reset

        self.moves = copy.deepcopy(self.env.env.env.num_moves_list)
        self.moves_nodefrag = copy.deepcopy(self.env.env.env.num_moves_list)
        self.x_data = np.arange(-400, 0)

        self.update_rfrag()
        self.update_blocked()
        # self.update_reward()
        self.update_moves()
        self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation)

        # self.threadpool.waitForDone()

    def plot_topology(self, highlight=False, nodes=[]):
        figure = plt.figure()
        sc = FigureCanvasQTAgg(figure)
        # Plot the NetworkX graph on the Matplotlib canvas
        G = self.env.env.env.topology
        pos = nx.spring_layout(G, seed=20)  # You can choose a layout algorithm here
        # pos = nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1000)

        if highlight:
            # nodes = self.env.env.env.previous_service.route.node_list
            for i in range(len(nodes) - 1):
                nx.draw_networkx_edges(G, pos, edgelist=[(nodes[i], nodes[i + 1])], edge_color='red', width=2)

        return sc

    def update_rfrag(self, r_frag_update=[], r_frag_nodefrag_update=[]):
        self.rfrag_plot.figure.clf()
        ax = self.rfrag_plot.figure.gca()
        temporarily_array = self.r_frag[len(r_frag_update):]
        temporarily_array.extend(r_frag_update)
        self.r_frag = temporarily_array
        temporarily_nodefrag_array = self.r_frag_nodefrag[len(r_frag_nodefrag_update):]
        temporarily_nodefrag_array.extend(r_frag_nodefrag_update)
        self.r_frag_nodefrag = temporarily_nodefrag_array
        ax.plot(self.x_data, self.r_frag, label='DRL', color='blue')
        ax.plot(self.x_data, self.r_frag_nodefrag, label='No defragmentation', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("RSS metric")
        ax.set_title("RSS metric")
        ax.legend()
        self.rfrag_plot.draw()

    # def plot_noc(self):
    #     # x = np.arange(-400, 0)
    #     figure = plt.figure(figsize=(15, 10))
    #     sc = FigureCanvasQTAgg(figure)
    #     fig = sc.figure
    #     ax = fig.add_subplot(111)
    #     ax.plot(self.x_data , self.env.env.env.num_cut_list_after, label='NoC metric', color='blue')
    #     ax.plot(self.x_data, self.env_no_df.env.env.num_cut_list_after, label='NoC metric', color='red')
    #
    #     ax.set_xlabel("Time unit")
    #     ax.set_ylabel("NoC metric")
    #     ax.set_title("NoC metric")
    #     ax.legend()
    #     sc.draw()
    #     return sc

    def update_blocked(self, blocked_update=[], blocked_nodefrag_update=[], b=0, c=0):
        self.blocked_plot.figure.clf()
        ax = self.blocked_plot.figure.gca()
        temporarily_array = self.blocked[len(blocked_update):]
        temporarily_array.extend(blocked_update)
        self.blocked = temporarily_array
        temporarily_nodefrag_array = self.blocked_nodefrag[len(blocked_nodefrag_update):]
        temporarily_nodefrag_array.extend(blocked_nodefrag_update)
        self.blocked_nodefrag = temporarily_nodefrag_array
        ax.plot(self.x_data, self.blocked, label='DRL', color='blue')
        ax.plot(self.x_data, self.blocked_nodefrag, label='No defragmentation', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of blocked services")
        ax.set_title("Number of blocked services")
        ax.legend()
        self.blocked_plot.draw()

    def update_moves(self, moves_update=[], moves_nodefrag_update=[]):
        self.moves_plot.figure.clf()
        ax = self.moves_plot.figure.gca()

        temporarily_array = self.moves[len(moves_update):]
        temporarily_array.extend(moves_update)
        self.moves = temporarily_array
        temporarily_nodefrag_array = self.moves_nodefrag[len(moves_nodefrag_update):]
        temporarily_nodefrag_array.extend(moves_nodefrag_update)
        self.moves_nodefrag = temporarily_nodefrag_array

        ax.plot(self.x_data, self.moves, label='DRL', color='blue')
        ax.plot(self.x_data, self.moves_nodefrag, label='No defragmentation', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of connection reallocation")
        ax.set_title("Number of connection reallocation")
        ax.legend()
        self.moves_plot.draw()

    # I tried to have one function for updating all figures, but I did not have time to fix the bug
    # def update_figures(self, fig, y_data, y_data_nodefrag, y_update = [], y_update_nodefrag= [], title= None):
    #     fig.figure.clf()
    #     ax = fig.figure.gca()
    #
    #     temporarily_array = y_data[len(y_update):]
    #     temporarily_array.extend(y_update)
    #     y_data = temporarily_array
    #     temporarily_nodefrag_array = y_data_nodefrag[len(y_update_nodefrag):]
    #     temporarily_nodefrag_array.extend(y_update_nodefrag)
    #     y_data_nodefrag = temporarily_nodefrag_array
    #
    #     ax.plot(self.x_data, y_data, label='DRL', color='blue')
    #     ax.plot(self.x_data, y_data_nodefrag, label='No defragmentation', color='red')
    #     ax.set_xlabel("Time unit")
    #     ax.set_ylabel(title)
    #     ax.set_title(title)
    #     ax.legend()
    #     fig.draw()

    def plot_figure(self, y_data, y_data_nodefrag, y_title):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        ax.plot(self.x_data, y_data, label='DRL', color='blue')
        ax.plot(self.x_data, y_data_nodefrag, label='No defragmentation', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel(y_title)
        ax.set_title(y_title)
        ax.legend()
        sc.draw()
        return sc

    def plot_reward(self):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        sum_rewards = []
        cumulative_sum = 0
        for i, value in enumerate(self.env.env.env.rewards):
            cumulative_sum += value
            sum_rewards.append(cumulative_sum)

        ax.plot(self.x_data, sum_rewards, label='DRL', color='blue')
        # ax.plot(self.x_data, sum_rewards, label='Sum of rewards for No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        sc.draw()
        return sc

    # def update_reward(self, reward_update=[], reward_nodefrag_update=[]):
    #     self.reward_plot.figure.clf()
    #     ax = self.reward_plot.figure.gca()
    #     reward_update_array = np.array(reward_update)
    #     temporarily_array = self.reward[len(reward_update_array):]
    #     self.reward = np.concatenate((temporarily_array, reward_update_array))
    #     reward_update_array_nodefrag = np.array(reward_nodefrag_update)
    #     temporarily_nodefrag_array = self.reward_nodefrag[len(reward_update_array_nodefrag):]
    #     self.reward_nodefrag = np.concatenate((temporarily_nodefrag_array, reward_update_array_nodefrag))
    #
    #     sum_rewards = []
    #     cumulative_sum = 0
    #     for i, value in enumerate(self.reward):
    #         cumulative_sum += value
    #         sum_rewards.append(cumulative_sum)
    #     sum_rewards_nodefrag = []
    #     cumulative_sum_nodefrag = 0
    #     for i, value in enumerate(self.reward_nodefrag):
    #         cumulative_sum_nodefrag += value
    #         sum_rewards_nodefrag.append(cumulative_sum)
    #
    #     if len(sum_rewards) > 400:
    #         a = 1
    #     ax.plot(self.x_data, sum_rewards, label=' DRL', color='blue')
    #     ax.plot(self.x_data, sum_rewards_nodefrag, label=' No Defrag', color='red')
    #     ax.set_xlabel("Time unit")
    #     ax.set_ylabel("Reward")
    #     ax.set_title("Reward")
    #     ax.legend()
    #     self.reward_plot.draw()

    def plot_grid(self, drl=True, first_allocation=False):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        topology = self.env.env.env.topology
        if first_allocation:
            # slot_allocation = self.env.env.env.last_spectrum_slot_allocation
            slot_allocation = self.slot_allocation_before_action
            title = "Spectrum Assignment before connection reallocation"
        else:
            if drl:
                slot_allocation = self.env.env.env.spectrum_slots_allocation
                title = "Spectrum Assignment"
            else:
                slot_allocation = self.env_no_df.env.env.spectrum_slots_allocation
                title = "Spectrum Assignment for No defragmentation scenario"
        # Plot the spectrum assignment graph
        return plot_spectrum_assignment_on_canvas(topology, slot_allocation, sc, values=False,
                                                  title=title)

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




class DtMainWindow(QMainWindow):
    """
    Responsibilities:
    - call functions
    - all plotting
    """

    def __init__(self, tapi_client):
        super().__init__()
        self.tapi_client = tapi_client

        self.drl_stop = True  ## I define this to control behaviour of continue defragmentation
        self.topology = None
        self.slot_allocation = None

        self.threadpool = QThreadPool()

        self.setWindowTitle("DefragDashboard for digital twin")
        pagelayout = QVBoxLayout()
        topology_layout = QHBoxLayout()
        button_and_text_layout = QHBoxLayout()  # Use QHBoxLayout for buttons and QTextEdit together.

        # self.load()  # TODO: comment later

        text = self.load_topology()
        self.slot_allocation = np.full((1, 1), fill_value=-1)

        self.topology_plot = self.dt_plot_topology()
        self.topology_plot.setFixedSize(400, 500)
        topology_layout.addWidget(self.topology_plot)
        self.grid_plot = self.dt_plot_grid()
        self.grid_plot.setFixedSize(1000, 500)
        topology_layout.addWidget(self.grid_plot)

        pagelayout.addLayout(topology_layout)
        pagelayout.addLayout(button_and_text_layout)

        # Create button panel
        button_panel = QWidget()
        button_panel_layout = QVBoxLayout()
        button_panel.setLayout(button_panel_layout)
        button_load = QPushButton("Load")
        button_load.pressed.connect(self.load)
        button_defrag = QPushButton("Perform Defragmentation")
        button_defrag.pressed.connect(self.perform_defragmentation)
        button_reset = QPushButton("Reset")
        button_reset.pressed.connect(self.reset_exp)
        button_panel_layout.addWidget(button_load)
        button_panel_layout.addWidget(button_defrag)
        button_panel_layout.addWidget(button_reset)

        # Create QTextEdit

        # path_create = "tapi/tapi-create.json"
        # path_delete = "tapi/tapi-delete.json"
        # json_data = load_json_from_file(path_create)
        # json_data_delete = load_json_from_file(path_delete)
        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(10, 10, 280, 380)
        # text_edit.setPlainText(json.dumps(json_data, indent=4))
        self.text_edit.setReadOnly(True)
        self.text_edit.setMarkdown(text)
        # Add button panel and QTextEdit to button_and_text_layout
        button_and_text_layout.addWidget(button_panel)
        button_and_text_layout.addWidget(self.text_edit)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)
        self.showFullScreen()
    
    def load_topology(self):
        self.topology = nx.Graph()
        self.node_dict = {}
        text = ""
        
        context = self.tapi_client.get_context()
        text += "---\n## Load topology\n"
        text += "### HTTP GET https://localhost:8082/restconf/data/tapi-common:context\n\n"
        text += f"```json\n{json.dumps(context, indent=2)}\n```"
        for node in context["tapi-common:context"]["tapi-topology:topology-context"]["topology"][0]["node"]:
            name = node["name"][0]["value"]
            if "." in name:
                name = name.split(".")[0]
            self.topology.add_node(name, uuid=node["uuid"], sip=[])
            self.node_dict[node["uuid"]] = name

            # loading the SIPs
            for nep in node["owned-node-edge-point"]:
                # skipping disabled end points
                if len(nep["mapped-service-interface-point"]) == 0 \
                    or nep["layer-protocol-name"] == "ODU" \
                    or nep["layer-protocol-name"] == "PHOTONIC_MEDIA" \
                    or "tapi-dsr:DIGITAL_SIGNAL_TYPE_100_GigE" not in nep["supported-cep-layer-protocol-qualifier"] \
                    or "tapi-dsr:DIGITAL_SIGNAL_TYPE_OTU_4" not in nep["supported-cep-layer-protocol-qualifier"]:
                    # or nep['lifecycle-state'] != 'INSTALLED':
                    # or nep["administrative-state"] != 'UNLOCKED' \
                    # or nep['operational-state'] != 'ENABLED':
                    continue
                for sip in nep["mapped-service-interface-point"]:
                    self.topology.nodes[name]["sip"].append(sip["service-interface-point-uuid"])
        
        _id = 0
        for link in context["tapi-common:context"]["tapi-topology:topology-context"]["topology"][0]["link"]:
            
            found = False
            for name in link["name"]:
                if "NODE" in name["value"] and "OPTICAL" not in name["value"] and "ODU" not in name["value"] and "OTN" not in name["value"]:
                    found = True
                    break
            if not found:
                continue

            node_1 = self.node_dict[link["node-edge-point"][0]["node-uuid"]]
            node_2 = self.node_dict[link["node-edge-point"][1]["node-uuid"]]
            if self.topology.has_node(node_1) \
                    and self.topology.has_node(node_2) \
                    and not self.topology.has_edge(node_1, node_2):
                # print(_id, link["uuid"])
                # print("\t", node_1, node_2)
                self.topology.add_edge(node_1, node_2, id=_id, uuid=link["uuid"])
                _id += 1
        return text
    
    def perform_defragmentation(self):
        response = requests.delete(
            f"https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/tapi-connectivity:connectivity-service={self.service_to_defragment}/",
            headers={
                "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0",
            },
            verify=False,
            timeout=300,
        )
        print(response.status_code)
        text = "---\n## Performing defragmentation\n"
        text += f"### Deleting service {self.service_to_defragment}\n"
        text += f"#### HTTP DELETE https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/tapi-connectivity:connectivity-service={self.service_to_defragment}/\n\n"
        if response.status_code != 204:
            text += "\n\n#### <p font-color='red'>Error!</p>"
            text += "\n\n"
            text += self.text_edit.toMarkdown()
            self.text_edit.setMarkdown(text)
            return
        else:
            text += "\n\n#### <p font-color='blue'>Success!</p>"
            text += "\n\n"
        
        # self.load()  # update visualization
        text += self.text_edit.toMarkdown()
        self.text_edit.setMarkdown(text)
        # self.text_edit.drawFrame()

        # create new service
        time.sleep(10)

        create = requests.post(
            "https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context",
            headers={
                "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0",
                "Content-Type": "application/yang-data+json",
            },
            data="""
            {

    "tapi-connectivity:connectivity-service": [

        {

            "end-point": [

                {

                    "layer-protocol-name": "DSR",

                    "layer-protocol-qualifier": "tapi-dsr:DIGITAL_SIGNAL_TYPE_100_GigE",

                    "service-interface-point": {

                        "service-interface-point-uuid": "00000010-0000-0000-0000-000000060549"

                    },

                    "tapi-adva:adva-connectivity-service-end-point-spec": {

                        "adva-network-port-parameters": {

                            "channel": {

                                "central-frequency": 19130

                            },

                            "termination-level": "OPU"

                        }

                    }

                },

                {

                    "layer-protocol-name": "DSR",

                    "layer-protocol-qualifier": "tapi-dsr:DIGITAL_SIGNAL_TYPE_100_GigE",

                    "service-interface-point": {

                        "service-interface-point-uuid": "00000010-0000-0000-0000-000000060578"

                    },

                    "tapi-adva:adva-connectivity-service-end-point-spec": {

                        "adva-network-port-parameters": {

                            "channel": {

                                "central-frequency": 19130

                            },

                            "termination-level": "OPU"

                        }

                    }

                }

            ],

            "service-layer": "DSR",

            "service-type": "POINT_TO_POINT_CONNECTIVITY",

            "name": [

                {

                    "value": "CFC_WCC100G_66_70_service_2",

                    "value-name": "USER"

                }

            ],

            "include-link": ["30000060-0000-0000-0000-000000048794"]

        }

    ]

}
            """,
            verify=False
        )
        print(create.status_code)
        text = f"---\n### Creating defragmented service\n"
        text += f"#### HTTP POST https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context\n\n"
        if create.status_code != 201:
            text += "\n\n#### <p font-color='red'>Error!</p>"
            text += "\n\n"
            text += self.text_edit.toMarkdown()
            self.text_edit.setMarkdown(text)
            return
        else:
            text += "\n\n#### <p font-color='blue'>Success!</p>"
            text += "\n\n"
        
        # self.load()  # update visualization
        text += self.text_edit.toMarkdown()
        self.text_edit.setMarkdown(text)

        time.sleep(5)


    def load(self):
        
        self.slot_allocation = np.full((self.topology.number_of_edges(), 6), fill_value=-1)

        services = self.tapi_client.get_services()
        service_mapping = {}
        for service in services['tapi-connectivity:connectivity-service']:
            found = False
            s_name = ""
            for name in service["name"]:
                # print(name)
                if "ODU" in name["value"] or "GBE100" in name["value"]:
                    continue
                found = True
                s_name = name["value"]
            if found:
                if s_name.endswith("_NFC"):
                    s_name = s_name.replace("_NFC", "")
                    if s_name not in service_mapping:
                        service_mapping[s_name] = {}
                    for endpoint in service["end-point"]:
                        service_mapping[s_name]["central-frequency"] = endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"]["channel"]["central-frequency"]
                else:
                    # pprint(service)
                    # pprint(service["end-point"])
                    if s_name not in service_mapping:
                        service_mapping[s_name] = {"uuid": service["uuid"]}
                    else:
                        service_mapping[s_name]["uuid"] = service["uuid"]
                    for endpoint in service["end-point"]:
                        for key, value in endpoint["connection-end-point"][0].items():
                            if key not in service_mapping[s_name]:
                                service_mapping[s_name][key] = []
                            service_mapping[s_name][key].append(value)
        print("\n\n", service_mapping, "\n\n")

        text = "---\n## Load services\n"
        text += "### HTTP GET https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/connectivity-service\n\n"
        text += f"```json\n{json.dumps(services, indent=2)}\n```"
        text += "\n\n"
        text += self.text_edit.toMarkdown()
        self.text_edit.setMarkdown(text)
        # self.text_edit.draw()

        _id = 1
        for service, values in service_mapping.items():
            print(service, values)
            # print(self.topology.edges())
            if service == "CFC_WCC100G_66_70_service_2" and "uuid" in values:
                self.service_to_defragment = values["uuid"]
                print("\n\nfound service to defragment", self.service_to_defragment)
                print(service)
                print(values)
                print("\n\n")
            freq = int(values["central-frequency"]) / 10000
            channel = int((freq - 19125) / 5)
            print("\tuuid:", values["uuid"])
            print("\tchannel:", values["central-frequency"])
            if "node-uuid" in values:
                source = self.node_dict[values["node-uuid"][0]]
                print("\tsource:", source, values["node-uuid"][0])
                destination = self.node_dict[values["node-uuid"][1]]
                print("\tdst:", destination, values["node-uuid"][1])
                link_id = self.topology[source][destination]["id"]
                print("\t", channel, link_id, _id)
            print("\n")
            self.slot_allocation[link_id, channel] = _id
            _id += 1

        # for i in range(10):
        #     index = randint(0, self.topology.number_of_edges() - 1)
        #     row = randint(0, 9)
        #     self.slot_allocation[index, row] = randint(1, 20)
        self.update_grid()
    
    def reset_exp(self):
        if hasattr(self, "service_to_defragment") and self.service_to_defragment != None:
            response = requests.delete(
                f"https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/tapi-connectivity:connectivity-service={self.service_to_defragment}/",
                headers={
                    "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0",
                },
                verify=False,
                timeout=300,
            )
            print("deleting for reset:", self.service_to_defragment, response.status_code)

            if response.status_code != 204:
                print("error deleting service!!!")
                # return

        # create new service
        time.sleep(10)

        create = requests.post(
            "https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context",
            headers={
                "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0",
                "Content-Type": "application/yang-data+json",
            },
            data="""
            {

    "tapi-connectivity:connectivity-service": [

        {

            "end-point": [

                {

                    "layer-protocol-name": "DSR",

                    "layer-protocol-qualifier": "tapi-dsr:DIGITAL_SIGNAL_TYPE_100_GigE",

                    "service-interface-point": {

                        "service-interface-point-uuid": "00000010-0000-0000-0000-000000060549"

                    },

                    "tapi-adva:adva-connectivity-service-end-point-spec": {

                        "adva-network-port-parameters": {

                            "channel": {

                                "central-frequency": 19145

                            },

                            "termination-level": "OPU"

                        }

                    }

                },

                {

                    "layer-protocol-name": "DSR",

                    "layer-protocol-qualifier": "tapi-dsr:DIGITAL_SIGNAL_TYPE_100_GigE",

                    "service-interface-point": {

                        "service-interface-point-uuid": "00000010-0000-0000-0000-000000060578"

                    },

                    "tapi-adva:adva-connectivity-service-end-point-spec": {

                        "adva-network-port-parameters": {

                            "channel": {

                                "central-frequency": 19145

                            },

                            "termination-level": "OPU"

                        }

                    }

                }

            ],

            "service-layer": "DSR",

            "service-type": "POINT_TO_POINT_CONNECTIVITY",

            "name": [

                {

                    "value": "CFC_WCC100G_66_70_service_2",

                    "value-name": "USER"

                }

            ],

            "include-link": ["30000060-0000-0000-0000-000000048794"]

        }

    ]

}
            """,
            verify=False
        )
        print("creating for reset:", create.status_code)
        if create.status_code != 201:
            print("error creating!")
            return
        self.slot_allocation = np.full((1, 1), fill_value=-1)
        self.update_grid()

    def dt_plot_topology(self):
        figure = plt.figure()
        sc = FigureCanvasQTAgg(figure)
        pos = nx.spring_layout(self.topology, seed=2)  # You can choose a layout algorithm here
        pos = {"NODE70": (1, 1), "NODE62": (1, 0), "NODE64": (0, 0), "NODE66": (0, 1)}
        nx.draw(self.topology, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1000)
        figure.gca().set_xlim([-.3, 1.3])
        figure.gca().set_ylim([-.3, 1.3])
        return sc

    def dt_plot_grid(self):
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        title = "Spectrum Assignment"
        return plot_spectrum_assignment_on_canvas(self.topology, self.slot_allocation, sc, values=True,
                                                  title=title)

    def update_grid(self):
        self.grid_plot.figure.clf()
        title = "Spectrum Assignment"
        canvas = plot_spectrum_assignment_on_canvas(self.topology, self.slot_allocation, self.grid_plot.figure.canvas, values=True,
                                                  title=title)
        canvas.draw()



def plot_spectrum_assignment_on_canvas(topology, vector, canvas, values=False, title=None):
    # Create a Matplotlib figure and use the provided canvas
    fig = canvas.figure
    fig.clf()
    ax = fig.add_subplot(111)

    # cmap = copy.copy(plt.cm.viridis)

    cmap = plt.cm.get_cmap("tab20")
    cmap.set_under(color='white')

    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')

    # p = ax.pcolor(vector, cmap=cmap, vmin=-0.0001, edgecolors='gray')
    if sum(vector[vector > -1]) == 0:
        return canvas
    masked_a = np.ma.masked_equal(vector, -1, copy=False)
    norm = mcolors.LogNorm(vmin=masked_a.min(), vmax=vector.max())

    p = ax.pcolor(vector, cmap=cmap, norm=norm, edgecolors='gray')


   #TODO: plotting a box between old initial slot and new slot when one connection is reallocated.
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
        ax.set_title(title)
    #     plt.yticks([topology.edges[link]['id']+.5 for link in topology.edges()], [link[0] + '-' + link[1] for link in topology.edges()])
    ax.set_yticks([topology.edges[link]['id'] + .5 for link in topology.edges()],
                  [f'{topology.edges[link]["id"]} ({link[0]}-{link[1]})' for link in topology.edges()])
    #     plt.colorbar()
    ax.set_xticks([x + 0.5 for x in plt.xticks()[0][:-1]], [x for x in plt.xticks()[1][:-1]])
    plt.tight_layout()
    # canvas.draw()
    return canvas


def load_json_from_file(filename):
    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
