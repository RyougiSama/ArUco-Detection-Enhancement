import sys
import time

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# --- 1. Worker (工作对象) ---
# 负责执行耗时任务，必须继承自 QObject
class Worker(QObject):
    # 定义信号，用于向主线程发送进度和完成通知
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    # 槽函数：这是在新线程中执行的入口
    def run_long_task(self):
        print(f"Worker: 任务开始在线程ID {QThread.currentThreadId()}")
        # 模拟耗时操作：从 0 计数到 10
        for i in range(11):
            time.sleep(0.3)  # 暂停 0.3 秒模拟工作

            progress = i * 10
            # 发射信号，通知主线程进度
            self.progress_signal.emit(progress)

            # 允许在任务执行期间退出线程（可选）
            current_thread = QThread.currentThread()
            if current_thread is not None and current_thread.isInterruptionRequested():
                print("Worker: 任务被请求中断。")
                break

        print(f"Worker: 任务结束在线程ID {QThread.currentThreadId()}")
        self.finished_signal.emit()


# --- 2. MainWindow (主窗口/控制器) ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 QThread 示例")
        self.setGeometry(100, 100, 400, 200)

        # UI 组件
        self._start_button = QPushButton("开始耗时任务")
        self._progress_label = QLabel("进度: 等待开始...")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self._start_button)
        layout.addWidget(self._progress_label)
        self.setLayout(layout)

        # 定义属性
        self._thread = None
        self._worker = None

        # 连接信号
        self._start_button.clicked.connect(self._start_task)

        # 显示主线程ID
        print(f"主线程 (GUI) ID: {QThread.currentThreadId()}")

    def _start_task(self):
        if self._thread is not None and self._thread.isRunning():
            print("任务已经在运行中...")
            return

        # 1. 创建 QThread 对象（线程执行流）
        self._thread = QThread()
        # 2. 创建 Worker 对象（包含要执行的逻辑）
        self._worker = Worker()

        # 3. **将 Worker 移动到新线程**
        self._worker.moveToThread(self._thread)

        # --- 连接信号和槽 ---

        # 4. 核心启动连接：线程开始 -> Worker 的 run_long_task 槽
        self._thread.started.connect(self._worker.run_long_task)

        # 5. 进度报告：Worker 信号 -> 主线程的槽函数 (更新 UI)
        self._worker.progress_signal.connect(self._update_progress)

        # 6. 完成清理：Worker 结束信号 -> 停止线程并清理资源
        self._worker.finished_signal.connect(self._thread.quit)
        self._worker.finished_signal.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._reset_thread)

        # 7. 更新 UI 状态并启动线程
        self._start_button.setEnabled(False)
        self._progress_label.setText("进度: 0% (任务运行中...)")
        self._thread.start()
        print("主线程: 启动 QThread...")

    def _update_progress(self, progress):
        """此槽函数在主线程中执行，安全地更新 UI"""
        self._progress_label.setText(f"进度: {progress}%")

        if progress >= 100:
            self._start_button.setEnabled(True)
            self._progress_label.setText("进度: 任务完成！")

    def _reset_thread(self):
        self._worker = self._thread = None


# --- 3. 应用程序入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    print(f"Env Args: {sys.argv}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
