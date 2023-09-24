import copy
import itertools
import pickle
from random import randint
import json

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QTextEdit
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

        title_label = QLabel("Create a connectivity service TAPI message")
        font = QFont()
        font.setPointSize(16)  # Set the desired font size for the title
        title_label.setFont(font)

        create_message.addWidget(title_label)
        sd= [self.main_window.env.env.env.last_service_to_defrag.source, self.main_window.env.env.env.last_service_to_defrag.destination]
        central_freq =  self.main_window.env.env.env.last_service_to_defrag.initial_slot + self.main_window.env.env.env.last_service_to_defrag.number_slots
        for i, endpoint in enumerate(json_data["tapi-connectivity:connectivity-service"][0]["end-point"]):
            endpoint["service-interface-point"]["service-interface-point-uuid"] = sd[i]
            endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"]["channel"][
                "central-frequency"] = f"{central_freq}"
            endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"][
                "rx-channel"]["central-frequency"] = f"{central_freq}"

        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(10, 10, 580, 380)
        self.text_edit.setPlainText(json.dumps(json_data, indent=4))
        create_message.addWidget(self.text_edit)

        label_text = f"demand ID is {self.main_window.env.env.env.last_service_to_defrag.service_id} \n from source {self.main_window.env.env.env.last_service_to_defrag.source} to destination {self.main_window.env.env.env.last_service_to_defrag.destination}. \n The route" \
                     f"is {self.main_window.env.env.env.last_service_to_defrag.route.node_list} and \n the old initial slot is {self.main_window.env.env.env.last_old_initial_slot} and \n the new initial slot is {self.main_window.env.env.env.last_new_initial_slot}"
        label = QLabel(label_text)
        font = QFont()
        font.setPointSize(16)  # Set the desired font size
        label.setFont(font)
        label.setFixedSize(400, 500)
        delete_message.addWidget(label)


        title_label = QLabel("Delete a connectivity service TAPI message")
        font = QFont()
        font.setPointSize(16)  # Set the desired font size for the title
        title_label.setFont(font)

        delete_message.addWidget(title_label)
        json_data_delete["tapi-connectivity:input"]["tapi-connectivity:service-id-or-name"] = self.main_window.env.env.env.last_service_to_defrag.service_id
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlainText(json.dumps(json_data_delete, indent=4))

        delete_message.addWidget(self.text_edit)

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

        label_text = f"demand ID is {self.main_window.env.env.env.last_service_to_defrag.service_id} \n from source {self.main_window.env.env.env.last_service_to_defrag.source} to destination {self.main_window.env.env.env.last_service_to_defrag.destination}. \n The route" \
                     f"is {self.main_window.env.env.env.last_service_to_defrag.route.node_list} and \n the old initial slot is {self.main_window.env.env.env.last_old_initial_slot} and \n the new initial slot is {self.main_window.env.env.env.last_new_initial_slot}"
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

        self.rfrag_plot = self.main_window.plot_rfrag()
        first_layout.addWidget(self.rfrag_plot)

        self.shanon_plot = self.main_window.plot_shanon()
        first_layout.addWidget(self.shanon_plot)

        # self.noc_plot = self.main_window.plot_noc()
        # first_layout.addWidget(self.shanon_plot)

        self.reward_plot = self.main_window.plot_reward()
        first_layout.addWidget(self.reward_plot)
        #
        self.blocked_plot = self.main_window.plot_blocked()
        second_layout.addWidget(self.blocked_plot)

        self.moves_plot = self.main_window.plot_moves()
        second_layout.addWidget(self.moves_plot)

        self.cycles_plot = self.main_window.plot_cycles()
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
        self.blocked = env.env.env.blocked_services[-400:]
        self.blocked_nodefrag = env.env.env.blocked_services[-400:]
        self.r_frag_nodefrag = env.env.env.rfrag_after_list[-400:]
        self.r_frag = env.env.env.rfrag_after_list[-400:]
        self.reward = env.env.env.rewards[-400:]
        self.reward_nodefrag = env.env.env.rewards[-400:]
        self.moves = env.env.env.num_moves_list[-400:]
        self.moves_nodefrag = env.env.env.num_moves_list[-400:]
        # self.cycles = env.env.env.defragmentation_procedure_list[-400:]
        # self.cycles_nodefrag = env.env.env.defragmentation_procedure_list[-400:]
        self.env = env
        self.env_no_df = copy.deepcopy(env)
        self.saved_env = copy.deepcopy(env)
        self.agent = agent
        self.tapi_client = tapi_client
        self.w = None  # No external window yet.
        self.continue_flag = False  # to Continue the DRL-based defragmentation.
        self.worker_cnt = None
        self.x_data = np.arange(-400, 0)
        self.rfrag_total = []
        self.rftag_ni_total = []
        self.x= []
        self.y = []
        self.slot_allocation_before_action = None

        self.drl_stop = True ## I define this to control behaviour of continue defragmentation

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

        self.rfrag_plot = self.plot_rfrag()
        metric_layout.addWidget(self.rfrag_plot)
        # self.shanon_plot = self.plot_shanon()
        # metric_layout.addWidget(self.shanon_plot)
        #
        self.blocked_plot = self.plot_blocked()
        metric_layout.addWidget(self.blocked_plot)

        # self.reward_plot = self.plot_reward()
        # metric_layout.addWidget(self.reward_plot)

        self.moves_plot = self.plot_moves()
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
                # print(r_frag_update, "\n", r_frag_nodefrag_update)
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

        self.blocked = self.env.env.env.blocked_services[-400:]
        self.blocked_nodefrag = self.env.env.env.blocked_services[-400:]
        self.r_frag_nodefrag = self.env.env.env.rfrag_after_list[-400:]
        self.r_frag = self.env.env.env.rfrag_after_list[-400:]
        self.reward = self.env.env.env.rewards[-400:]
        self.reward_nodefrag = self.env.env.env.rewards[-400:]
        self.drl_stop = True
        self.continue_flag = False  # to Continue the DRL-based defragmentation. It should be reset

        self.moves = self.env.env.env.num_moves_list[-400:]
        self.moves_nodefrag = self.env.env.env.num_moves_list[-400:]
        self.x_data = np.arange(-400, 0)

        self.update_rfrag()
        self.update_blocked()
        # self.update_reward()
        self.update_moves()
        self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation)

        # self.threadpool.waitForDone()



    def plot_topology(self, highlight=False, nodes = []):
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



    def plot_rfrag(self):

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

    def update_rfrag(self, r_frag_update=[], r_frag_nodefrag_update=[]):
        self.rfrag_plot.figure.clf()
        ax = self.rfrag_plot.figure.gca()

        # print(r_frag_update, "\n", r_frag_nodefrag_update)
        temporarily_array = self.r_frag[len(r_frag_update):]

        # print(f"\n\t{r_frag_update}\n\t{r_frag_nodefrag_update}")
        temporarily_array.extend(r_frag_update)

        self.x.extend(r_frag_update)
        self.y.extend(r_frag_nodefrag_update)


        self.r_frag = temporarily_array


        # r_frag_update_array_nodefrag = np.array(r_frag_nodefrag_update)


        temporarily_nodefrag_array = self.r_frag_nodefrag[len(r_frag_nodefrag_update):]
        temporarily_nodefrag_array.extend(r_frag_nodefrag_update)

        self.r_frag_nodefrag = temporarily_nodefrag_array

        ax.plot(self.x_data, self.r_frag, label='RSS metric DRL', color='blue')
        ax.plot(self.x_data, self.r_frag_nodefrag, label='RSS metric No Defrag', color='red')

        count = 0


        # self.rfrag_total.extend(self.r_frag)
        # self.rftag_ni_total.extend(self.r_frag_nodefrag)
        # Make sure both lists have the same length
        # if len(self.x) == len(self.y):
        #     # Iterate through the lists using a for loop
        #     for i in range(len(self.y)):
        #         # Compare elements at the same index in both lists
        #         if self.y[i] in self.x:
        #             count += 1
        #     print(f"len of total rfrag: {len(self.x)}")
        #     print(f"Number of equal elements: {count}")
        # else:
        #     print("The lists have different lengths, so you cannot compare them el")


        ax.set_xlabel("Time unit")
        ax.set_ylabel("RSS metric")
        ax.set_title("RSS metric")
        ax.legend()
        self.rfrag_plot.draw()

    def plot_shanon(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(self.x_data , self.env.env.env.shanon_entrophy_after_list[-400:], label='Shanon metric')
        ax.plot(self.x_data, self.env_no_df.env.env.shanon_entrophy_after_list[-400:], label='Shanon metric No defrag')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Shanon metric")
        ax.set_title("Shanon metric")
        ax.legend()
        sc.draw()
        return sc

    def plot_noc(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)
        ax.plot(self.x_data , self.env.env.env.num_cut_list_after[-400:], label='NoC metric', color='blue')
        ax.plot(self.x_data, self.env_no_df.env.env.num_cut_list_after[-400:], label='NoC metric', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("NoC metric")
        ax.set_title("NoC metric")
        ax.legend()
        sc.draw()
        return sc

    def plot_blocked(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        ax.plot(self.x_data , self.blocked, label='Sum of blocked services DRL', color='blue')
        ax.plot(self.x_data , self.blocked_nodefrag, label='Sum of blocked services No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of blocked services")
        ax.set_title("Number of blocked services")
        ax.legend()
        sc.draw()
        return sc

    def update_blocked(self, blocked_update=[], blocked_nodefrag_update=[], b=0, c=0):
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
        ax.set_ylabel("Number of blocked services")
        ax.set_title("Number of blocked services")
        ax.legend()
        self.blocked_plot.draw()


    def plot_moves(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        ax.plot(self.x_data , self.moves, label='Sum of blocked services DRL', color='blue')
        ax.plot(self.x_data , self.moves_nodefrag, label='Sum of blocked services No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of connection reallocations")
        ax.set_title("Number of connection reallocations")
        ax.legend()
        sc.draw()
        return sc


    def update_moves(self, moves_update=[], moves_nodefrag_update=[]):
        self.moves_plot.figure.clf()
        ax = self.moves_plot.figure.gca()


        # blocked_update_array = np.array(blocked_update)
        # blocked_update_array += self.blocked[len(self.blocked)-1]

        temporarily_array = self.moves[len(moves_update):]
        # self.blocked = np.concatenate((temporarily_array, blocked_update_array))
        temporarily_array.extend(moves_update)
        self.moves = temporarily_array
        # blocked_update_array_nodefrag = np.array(blocked_nodefrag_update)

        # blocked_update_array_nodefrag += self.blocked_nodefrag[len(self.blocked_nodefrag)-1]
        temporarily_nodefrag_array = self.moves_nodefrag[len(moves_nodefrag_update):]
        temporarily_nodefrag_array.extend(moves_nodefrag_update)
        # self.blocked_nodefrag = np.concatenate((temporarily_nodefrag_array, blocked_update_array_nodefrag))
        self.moves_nodefrag = temporarily_nodefrag_array

        ax.plot(self.x_data, self.moves, label='Number of connection reallocation DRL', color='blue')
        ax.plot(self.x_data, self.moves_nodefrag, label='Number of connection reallocation No Defrag', color='red')
        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of connection reallocation")
        ax.set_title("Number of connection reallocation")
        ax.legend()
        self.moves_plot.draw()



    def plot_cycles(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        ax.plot(self.x_data , self.env.env.env.defragmentation_procedure_list[-400:], label='Number of defragmentation cycles', color='blue')
        ax.plot(self.x_data , self.env_no_df.env.env.defragmentation_procedure_list[-400:], label='Number of defragmentation cycles', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Number of defragmentation cycles")
        ax.set_title("Number of defragmentation cycles")
        ax.legend()
        sc.draw()
        return sc


    def plot_reward(self):
        # x = np.arange(-400, 0)
        figure = plt.figure(figsize=(15, 10))
        sc = FigureCanvasQTAgg(figure)
        fig = sc.figure
        ax = fig.add_subplot(111)

        sum_rewards = []
        cumulative_sum = 0
        for i, value in enumerate(self.env.env.env.rewards[-400:]):
            cumulative_sum += value
            sum_rewards.append(cumulative_sum)

        ax.plot(self.x_data, sum_rewards, label='Sum of rewards for DRL', color='blue')
        # ax.plot(self.x_data, sum_rewards, label='Sum of rewards for No Defrag', color='red')

        ax.set_xlabel("Time unit")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        sc.draw()
        return sc

    def update_reward(self, reward_update=[], reward_nodefrag_update=[]):
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
        return plot_spectrum_assignment_on_canvas(topology, slot_allocation, sc, values=True,
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

    # def step(self):
    #     action = self.agent.predict(env.observation())
    #     self.env.step(action)


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


def load_json_from_file(filename):
    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
