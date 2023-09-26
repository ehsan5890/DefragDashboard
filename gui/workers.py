import traceback
import sys
import copy

from PyQt6.QtCore import pyqtSignal, QRunnable, pyqtSlot, QObject


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


class RunArrivalsWorker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, environment_drl, environment_nodefrag, agent, drl_stop=True, drl_first_action=True):
        super(RunArrivalsWorker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.environment_drl = environment_drl
        self.environment_nodefrag = environment_nodefrag
        self.agent = agent
        self.drl_stop = drl_stop
        self.drl_first_action = drl_first_action

        # TODO: load the agent

        self.signals = WorkerSignals()
        self.stopped = False

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # while not self.stopped:

        flag_blocking = False
        flag_first_allocation = False
        r_frag_update = []
        r_frag_update_nodefrag = []
        reward_update = []
        reward_update_nodefrag = []
        blocked_update = []
        blocked_update_nodefrag = []
        moves_update = []
        moves_update_nodefrag = []
        for _ in range(50):
            if flag_blocking is True and self.drl_stop:
                break

            if self.drl_first_action and flag_first_allocation:
                break
            obs_drl = self.environment_drl.reset()
            obs_no_df = self.environment_nodefrag.reset()
            done, state = False, None
            steps = 0
            while not done:
                action, _states = self.agent.predict(obs_drl, deterministic=True)
                slot_allocation_before_action = copy.deepcopy(self.environment_drl.env.env.spectrum_slots_allocation)
                obs_drl, reward, done, info = self.environment_drl.step(action)
                if action == 0:
                    obs_no_df, reward_df, done_df, info_df = self.environment_nodefrag.step(0)

                    # r_frag_list = self.environment_drl.env.env.rfrag_after_list
                    # r_frag_update.append(r_frag_list[len(r_frag_list) - 1])
                    #
                    # r_frag_nodefrag_list = self.environment_nodefrag.env.env.rfrag_after_list
                    # r_frag_update_nodefrag.append(r_frag_nodefrag_list[len(r_frag_nodefrag_list) - 1])
                    #
                    # reward_list = self.environment_drl.env.env.rewards
                    # reward_update.append(reward_list[len(reward_list) - 1])
                    #
                    # reward_list_nodefrag = self.environment_nodefrag.env.env.rewards
                    # reward_update_nodefrag.append(reward_list_nodefrag[len(reward_list_nodefrag) - 1])
                    #
                    # blocked_list = self.environment_drl.env.env.blocked_services
                    # blocked_update.append(blocked_list[len(blocked_list) - 1])
                    #
                    # blocked_list_nodefrag = self.environment_nodefrag.env.env.blocked_services
                    # blocked_update_nodefrag.append(blocked_list_nodefrag[len(blocked_list_nodefrag) - 1])
                    #
                    # steps += 1
                elif reward == -1:

                    obs_drl, reward, done_2, info = self.environment_drl.step(0)
                    obs_no_df, reward_df, done_df, info_df = self.environment_nodefrag.step(0)
                elif action != 0 and self.drl_first_action:
                    flag_first_allocation = True
                    self.signals.result.emit((self.environment_drl, self.environment_nodefrag,
                                              flag_blocking, r_frag_update,
                                              r_frag_update_nodefrag, reward_update,
                                              reward_update_nodefrag,
                                              blocked_update,
                                              blocked_update_nodefrag, moves_update, moves_update_nodefrag, True,
                                              slot_allocation_before_action, flag_first_allocation))
                    break

                r_frag_update.append(self.environment_drl.env.env.rfrag_after_list[-1])
                r_frag_update_nodefrag.append(self.environment_nodefrag.env.env.rfrag_after_list[-1])
                reward_update.append(self.environment_drl.env.env.rewards[-1])
                reward_update_nodefrag.append(self.environment_nodefrag.env.env.rewards[-1])
                blocked_update.append(self.environment_drl.env.env.blocked_services[-1])
                blocked_update_nodefrag.append(self.environment_nodefrag.env.env.blocked_services[-1])
                moves_update.append(self.environment_drl.env.env.num_moves_list[-1])
                moves_update_nodefrag.append(self.environment_nodefrag.env.env.num_moves_list[-1])
                # blocked_update_nodefrag.append(b)
                steps += 1

                if self.environment_nodefrag.env.env.previous_service_accepted == False and self.environment_drl.env.env.previous_service_accepted == True and self.drl_stop:
                    flag_blocking = True
                    self.signals.result.emit((self.environment_drl, self.environment_nodefrag,
                                              flag_blocking, r_frag_update,
                                              r_frag_update_nodefrag, reward_update,
                                              reward_update_nodefrag,
                                              blocked_update,
                                              blocked_update_nodefrag, moves_update, moves_update_nodefrag, False,
                                              slot_allocation_before_action, flag_first_allocation))
                    break

                if steps % 10 == 0:
                    # self.signals.result.emit((self.environment_drl, self.environment_nodegrag, rewards_drl, rewards_nodefrag))
                    if self.drl_stop:
                        self.signals.result.emit((self.environment_drl, self.environment_nodefrag,
                                                  flag_blocking, r_frag_update,
                                                  r_frag_update_nodefrag,
                                                  reward_update, reward_update_nodefrag,
                                                  blocked_update, blocked_update_nodefrag, moves_update,
                                                  moves_update_nodefrag, True, slot_allocation_before_action,
                                                  flag_first_allocation))

                    else:
                        self.signals.result.emit((self.environment_drl, self.environment_nodefrag,
                                                  flag_blocking, r_frag_update,
                                                  r_frag_update_nodefrag,
                                                  reward_update, reward_update_nodefrag,
                                                  blocked_update, blocked_update_nodefrag, moves_update,
                                                  moves_update_nodefrag, False, slot_allocation_before_action,
                                                  flag_first_allocation))
                    # print(f"{steps}\n\t{r_frag_update}\n\t{r_frag_update_nodefrag}")
                    r_frag_update = []
                    r_frag_update_nodefrag = []

                    reward_update = []
                    reward_update_nodefrag = []

                    blocked_update = []
                    blocked_update_nodefrag = []

                    moves_update = []
                    moves_update_nodefrag = []

                    if self.stopped:
                        return

    def stop(self):
        self.stopped = True
