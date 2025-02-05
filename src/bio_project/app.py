import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
from PyQt5.QtCore import Qt
from openslide import OpenSlide
from PIL import ImageQt
from PyQt5.QtGui import QPixmap
import subprocess


class InferenceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bioinformatics Inference GUI')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.label = QLabel('Trascina qui un file .tif o selezionalo manualmente')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('border: 2px dashed #aaa; padding: 20px;')
        self.layout.addWidget(self.label)

        self.button_select = QPushButton('Seleziona File')
        self.button_select.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.button_select)

        self.button_start = QPushButton('Start Inference')
        self.button_start.clicked.connect(self.start_inference)
        self.layout.addWidget(self.button_start)

        self.setLayout(self.layout)
        self.file_path = ''

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.tif'):
                self.file_path = file_path
                self.label.setText(f'File selezionato: {os.path.basename(file_path)}')

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Seleziona File .tif', '', 'TIF files (*.tif)')
        if file_path:
            self.file_path = file_path
            self.label.setText(f'File selezionato: {os.path.basename(file_path)}')

    def start_inference(self):
        if self.file_path:
            command = [
                'python3', 'main.py',
                '--source_dir', self.file_path
            ]
            subprocess.run(command)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())