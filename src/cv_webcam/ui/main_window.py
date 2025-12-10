from PyQt5.QtWidgets import QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # 设置主窗口属性
        self.setWindowTitle("我的第一个 PyQt 程序")
        self.setGeometry(100, 100, 1000, 800)  # (x, y, width, height)

        # 创建按钮
        self.button = QPushButton("点击我", self)
        self.button.setGeometry(450, 350, 100, 50)  # (x, y, width, height)
        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        print("按钮被点击了！")
