import traceback
import sys

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

    def __init__(self, environment_drl, environment_nodefrag):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.environment_drl = environment_drl
        self.environment_nodegrag = environment_nodefrag

        # TODO: load the agent

        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''


        for _ in range(50):
            obs_drl = self.environment_drl.reset()
            obs_no_df = self.environment_nodefrag.reset()
            done, state = False, None
            arrival = 0
            while not done:
                action, _states = self.agent.predict(obs_drl, deterministic=True)
                obs_drl, reward, done, info = self.env.step(action)
                if action == 0:
                    obs_no_df, reward_df, done_df, info_df = self.env_no_df.step(0)

                if self.env_no_df.env.env.service.accepted == False and self.env.env.env.service.accepted == True:
                    a = 1

                if self.env_no_df.env.env.service.accepted == True and self.env.env.env.service.accepted == False:
                    b = 1

                # self.update_grid(self.env.env.env.topology, self.env.env.env.spectrum_slots_allocation )
                if reward == -1:
                    obs_drl, reward, done_2, info = self.env.step(0)
                    obs_no_df, reward_df, done_df, info_df = self.env_no_df.step(0)

                arrival += 1
                if arrival % 10 == 0:
                    self.signals.result.emit((self.environment_drl, self.environment_nodegrag, rewards_drl, rewards_nodefrag))
