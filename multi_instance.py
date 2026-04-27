import sys
import threading
import time
import logging
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QPlainTextEdit, QSpinBox, QComboBox, QCheckBox,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import loadData
import auto_fetch
import data_package
from recognize import MONSTER_COUNT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='multi_instance.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

class DeviceInstance:
    def __init__(self, port):
        self.port = port
        self.serial = f"127.0.0.1:{port}"
        self.connector = loadData.AdbConnector(self.serial)
        self.auto_fetch = None
        self.status = "已停止"

    def start(self, game_mode, is_invest):
        try:
            self.connector.connect()
            if not self.connector.is_connected:
                self.status = "连接失败"
                return False
            
            self.auto_fetch = auto_fetch.AutoFetch(
                self.connector,
                game_mode,
                is_invest,
                update_prediction_callback=lambda x: None,
                update_monster_callback=lambda x: None,
                updater=lambda: None,
                start_callback=lambda: None,
                stop_callback=lambda: None,
                training_duration=-1,
                is_multi_instance=True
            )
            self.auto_fetch.start_auto_fetch()
            self.status = "正在运行"
            return True
        except Exception as e:
            self.status = f"错误: {str(e)}"
            return False

    def stop(self):
        if self.auto_fetch:
            self.auto_fetch.stop_auto_fetch()
        self.status = "已停止"

    def get_status_line(self):
        if not self.auto_fetch or not self.auto_fetch.auto_fetch_running:
            return f"[{self.serial:<15}] 状态: {self.status}"
        
        af = self.auto_fetch
        elapsed = time.time() - af.start_time if af.start_time else 0
        hours, remainder = divmod(elapsed, 3600)
        minutes, _ = divmod(remainder, 60)
        
        state_name = "未知"
        if hasattr(af, 'last_state') and af.last_state:
            # 这里的映射需要对应 auto_fetch.GameState
            state_mapping = {
                "MAIN_MENU": "主菜单",
                "MODE_SELECTION_UNSELECTED": "选择模式",
                "MODE_SELECTION_SELECTED": "已选模式",
                "PRE_BATTLE": "备战中",
                "IN_BATTLE": "对战中",
                "SETTLEMENT": "结算中",
                "FINISHED": "已完成",
                "UNKNOWN": "未知"
            }
            raw_name = af.last_state.name if hasattr(af.last_state, 'name') else str(af.last_state)
            state_name = state_mapping.get(raw_name, raw_name)
        
        return (f"[{self.serial:<15}] "
                f"状态: {state_name:<8} | "
                f"填写: {af.total_fill_count:<3} | "
                f"错误: {af.incorrect_fill_count:<3} | "
                f"预测: {af.current_prediction:.2f} | "
                f"时长: {int(hours)}小时{int(minutes)}分钟")

class MultiInstanceManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("铁鲨鱼多开自动化工具")
        self.setGeometry(100, 100, 900, 600)
        
        self.instances = {}
        self.init_ui()
        
        # 定时更新界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 配置区域
        settings_layout = QHBoxLayout()
        
        self.game_mode_combo = QComboBox()
        self.game_mode_combo.addItems(["单人", "30人"])
        settings_layout.addWidget(QLabel("模式:"))
        settings_layout.addWidget(self.game_mode_combo)
        
        self.invest_check = QCheckBox("自动投资")
        self.invest_check.setChecked(False)
        settings_layout.addWidget(self.invest_check)
        
        layout.addLayout(settings_layout)
        
        # 端口输入
        layout.addWidget(QLabel("ADB端口 (每行一个，例如 5555):"))
        self.ports_input = QPlainTextEdit()
        # 尝试读取历史端口
        try:
            if Path("multi_ports.txt").exists():
                self.ports_input.setPlainText(Path("multi_ports.txt").read_text())
        except:
            pass
        self.ports_input.setPlaceholderText("5555\n5557\n5559")
        self.ports_input.setFixedHeight(100)
        layout.addWidget(self.ports_input)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("全部启动")
        self.start_btn.clicked.connect(self.start_all)
        self.stop_btn = QPushButton("全部停止")
        self.stop_btn.clicked.connect(self.stop_all)
        self.package_btn = QPushButton("打包数据")
        self.package_btn.clicked.connect(self.package_data)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.package_btn)
        layout.addLayout(btn_layout)
        
        # 状态显示
        layout.addWidget(QLabel("运行状态:"))
        self.status_display = QPlainTextEdit()
        self.status_display.setReadOnly(True)
        # 使用等宽字体以保证列对齐
        font = QFont("Courier New", 10)
        if sys.platform == "win32":
            font = QFont("Consolas", 10)
        self.status_display.setFont(font)
        layout.addWidget(self.status_display)

    def start_all(self):
        # 保存当前端口配置
        try:
            Path("multi_ports.txt").write_text(self.ports_input.toPlainText())
        except:
            pass
            
        ports = [p.strip() for p in self.ports_input.toPlainText().split('\n') if p.strip()]
        game_mode = self.game_mode_combo.currentText()
        is_invest = self.invest_check.isChecked()
        
        def run_start():
            for port in ports:
                # 检查是否已经在运行
                is_running = False
                if port in self.instances:
                    inst = self.instances[port]
                    if inst.auto_fetch and inst.auto_fetch.auto_fetch_running:
                        is_running = True
                
                if not is_running:
                    instance = DeviceInstance(port)
                    if instance.start(game_mode, is_invest):
                        self.instances[port] = instance
                    # 每个实例启动后延迟 2 秒，避免 ADB 冲突
                    time.sleep(2.0)
        
        threading.Thread(target=run_start, daemon=True).start()
    
    def stop_all(self):
        for instance in self.instances.values():
            instance.stop()
        self.instances.clear()
        self.update_display()

    def package_data(self):
        try:
            zip_filename = data_package.package_data()
            if zip_filename and Path(zip_filename).exists():
                # 在文件浏览器中高亮显示文件
                subprocess.run(f'explorer /select,"{Path(zip_filename).absolute()}"')
                QMessageBox.information(self, "成功", f"数据已打包到 {zip_filename}")
            else:
                QMessageBox.warning(self, "警告", "没有找到可以打包的数据目录或打包失败。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打包数据时发生错误: {str(e)}")

    def update_display(self):
        lines = []
        # 获取输入框中的所有端口
        input_ports = [p.strip() for p in self.ports_input.toPlainText().split('\n') if p.strip()]
        
        any_running = False
        for port in input_ports:
            if port in self.instances:
                instance = self.instances[port]
                lines.append(instance.get_status_line())
                if instance.auto_fetch and instance.auto_fetch.auto_fetch_running:
                    any_running = True
            else:
                serial = f"127.0.0.1:{port}"
                lines.append(f"[{serial:<15}] 状态: 已停止")
        
        # 只有在全部停止状态下才能打包
        self.package_btn.setEnabled(not any_running)
                
        self.status_display.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiInstanceManager()
    window.show()
    sys.exit(app.exec())
