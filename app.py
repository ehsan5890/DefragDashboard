import argparse
import sys
from random import randint

from PyQt6.QtWidgets import QApplication

from demo import MainWindow


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO: create a parser for the arguments
    # - path to the environment pickle file, None creates a new environment
    # - path to the trained agent
    # example: https://github.com/carlosnatalino/python-simple-anycast-wdm-simulator/blob/8fda7f7b19aa092b15d46578f678d45e392b872a/run.py#L140C5-L140C39

    # TODO:
    # - create an Gym environment or load existing environment from pickle file
    env = None  # from the assets folder

    # load the agent from the assets folder
    agent = None

    app = QApplication(sys.argv)
    w = MainWindow(env, agent)  # TODO: pass the environment to the main window
    w.show()
    app.exec()
