from pytestqt.qtbot import QtBot

from demo import MainWindow


def test_main(qtbot: QtBot) -> None:
    widget: MainWindow = MainWindow(None, None)
    qtbot.addWidget(widget)

    widget.button.click()
