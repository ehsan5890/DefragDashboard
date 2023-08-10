from pytestqt.qtbot import QtBot

from gui import MainWindow


def test_main(qtbot: QtBot) -> None:
    widget: MainWindow = MainWindow(None, None, None)
    qtbot.addWidget(widget)

    widget.button.click()
