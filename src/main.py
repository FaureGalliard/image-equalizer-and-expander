import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QHBoxLayout,
    QComboBox, QSlider, QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from image_processor import load_image, compute_histogram, expand_histogram_custom, equalize_histogram_custom
from PIL import Image

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        self.initialize_variables()

    def setup_ui(self):
        self.setWindowTitle("Procesador de Histogramas Avanzado")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Controles superiores
        self.setup_top_controls(main_layout)
        
        # Ajustes del histograma
        self.setup_histogram_settings(main_layout)
        
        # Área de visualización
        self.setup_display_area(main_layout)

    def setup_top_controls(self, main_layout):
        controls_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Cargar Imagen")
        self.btn_load.setStyleSheet("background-color: #4CAF50; color: white;")
        
        self.btn_save = QPushButton("Guardar Imagen Procesada")
        self.btn_save.setStyleSheet("background-color: #FF9800; color: white;")
        
        self.btn_reset = QPushButton("Resetear Valores")
        self.btn_reset.setStyleSheet("background-color: #f44336; color: white;")
        
        self.btn_compare = QPushButton("Comparar Lado a Lado")
        
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(self.btn_save)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.btn_compare)
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)

    def setup_histogram_settings(self, main_layout):
        settings_layout = QFormLayout()
        
        # Controles para expansión
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 254)
        self.min_slider.setValue(0)
        self.min_slider.setTickInterval(10)
        self.min_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(1, 255)
        self.max_slider.setValue(255)
        self.max_slider.setTickInterval(10)
        self.max_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        # Controles para ecualización
        self.clip_limit_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_limit_slider.setRange(0, 50)  # 0-10%, con granularidad fina
        self.clip_limit_slider.setValue(0)  # 0.03 (3%)
        self.clip_limit_slider.setTickInterval(1)
        self.clip_limit_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        # Combobox para modo de histograma
        self.hist_mode = QComboBox()
        self.hist_mode.addItems(["Canales RGB", "Luminancia"])
        
        # Organización
        settings_layout.addRow("Modo Histograma:", self.hist_mode)
        settings_layout.addRow("Mínimo (Expansión):", self.min_slider)
        settings_layout.addRow("Máximo (Expansión):", self.max_slider)
        settings_layout.addRow("Límite Clip (Ecualización):", self.clip_limit_slider)
        
        main_layout.addLayout(settings_layout)

    def setup_display_area(self, main_layout):
        self.display_layout = QHBoxLayout()
        
        # Widget para imágenes
        self.image_label_original = QLabel()
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")
        
        self.image_label_processed = QLabel()
        self.image_label_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_processed.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")
        
        # Widget para histogramas
        self.canvas_original = FigureCanvas(Figure(figsize=(5, 3), dpi=100))
        self.canvas_original.setStyleSheet("background-color: white;")
        
        self.canvas_processed = FigureCanvas(Figure(figsize=(5, 3), dpi=100))
        self.canvas_processed.setStyleSheet("background-color: white;")
        
        self.display_layout.addWidget(self.image_label_original, 30)
        self.display_layout.addWidget(self.image_label_processed, 30)
        self.display_layout.addWidget(self.canvas_original, 20)
        self.display_layout.addWidget(self.canvas_processed, 20)
        
        main_layout.addLayout(self.display_layout, stretch=1)

    def setup_connections(self):
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_reset.clicked.connect(self.reset_values)
        self.btn_compare.clicked.connect(self.toggle_comparison)
        
        self.min_slider.valueChanged.connect(self.update_custom_expansion)
        self.max_slider.valueChanged.connect(self.update_custom_expansion)
        self.clip_limit_slider.valueChanged.connect(self.update_custom_equalization)
        self.hist_mode.currentTextChanged.connect(self.update_histogram_display)

    def initialize_variables(self):
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.compare_mode = False
        self.last_applied = None 

    def load_image(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Abrir Imagen",
                "",
                "Imágenes (*.png *.jpg *.jpeg)"
            )
            
            if path:
                self.current_image = load_image(path)
                self.original_image = self.current_image.copy()
                self.processed_image = None
                self.last_applied = None
                self.compare_mode = False
                self.btn_compare.setText("Comparar Lado a Lado")
                self.show_image(self.current_image, original=True)
                self.plot_histogram(compute_histogram(self.current_image, self.hist_mode.currentText()), original=True)
                self.clear_processed_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar imagen: {e}")

    def show_image(self, img_array, original=True):
        try:
            height, width = img_array.shape[0], img_array.shape[1]
            
            if len(img_array.shape) == 2:  # Escala de grises
                bytes_per_line = width
                qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:  # Color
                bytes_per_line = 3 * width
                img_array = np.ascontiguousarray(img_array[:, :, :3], dtype=np.uint8)
                qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage).scaled(
                400, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            if original:
                self.image_label_original.setPixmap(pixmap)
            else:
                self.image_label_processed.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al mostrar imagen: {e}")

    def plot_histogram(self, histogram, original=True):
        try:
            canvas = self.canvas_original if original else self.canvas_processed
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            
            if self.hist_mode.currentText() == "Luminancia" or not isinstance(histogram, list):
                hist_data = histogram
                color = 'gray'
                ax.bar(range(256), hist_data, color=color, width=1.0)
            else:  # Canales RGB
                colors = ['red', 'green', 'blue']
                for i, hist in enumerate(histogram):
                    ax.plot(hist, color=colors[i], alpha=0.7)
            
            ax.set_title("Histograma Original" if original else "Histograma Procesado")
            canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al graficar histograma: {e}")

    def toggle_comparison(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Advertencia", "Por favor, carga una imagen primero.")
            return
            
        self.compare_mode = not self.compare_mode
        self.btn_compare.setText("Volver a Vista Normal" if self.compare_mode else "Comparar Lado a Lado")
        
        if self.compare_mode:
            self.show_image(self.original_image, original=True)
            self.plot_histogram(compute_histogram(self.original_image, self.hist_mode.currentText()), original=True)
            if self.processed_image is not None:
                self.show_image(self.processed_image, original=False)
                self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=False)
            else:
                self.clear_processed_display()
        else:
            self.show_image(self.current_image, original=True)
            self.plot_histogram(compute_histogram(self.current_image, self.hist_mode.currentText()), original=True)
            self.clear_processed_display()

    def update_custom_expansion(self):
        if self.current_image is None:
            return
        try:
            self.processed_image = expand_histogram_custom(
                self.current_image,
                self.min_slider.value(),
                self.max_slider.value()
            )
            self.last_applied = "expansion"
            if self.compare_mode:
                self.show_image(self.processed_image, original=False)
                self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=False)
            else:
                self.show_image(self.processed_image, original=True)
                self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar expansión: {e}")

    def update_custom_equalization(self):
        if self.current_image is None:
            return
        try:
            self.processed_image = equalize_histogram_custom(
                self.current_image,
                self.clip_limit_slider.value() / 10000.0  # Mapear 0-1000 a 0.0-0.1
            )
            self.last_applied = "equalization"
            if self.compare_mode:
                self.show_image(self.processed_image, original=False)
                self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=False)
            else:
                self.show_image(self.processed_image, original=True)
                self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar ecualización: {e}")

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Advertencia", "No hay imagen procesada para guardar.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Imagen",
                "",
                "Imágenes (*.png *.jpg *.jpeg)"
            )
            if path:
                img = Image.fromarray(self.processed_image)
                img.save(path)
                QMessageBox.information(self, "Éxito", "Imagen guardada correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar imagen: {e}")

    def clear_processed_display(self):
        self.image_label_processed.clear()
        self.canvas_processed.figure.clear()
        self.canvas_processed.draw()

    def update_histogram_display(self):
        if self.original_image is not None:
            self.plot_histogram(compute_histogram(self.original_image, self.hist_mode.currentText()), original=True)
        if self.processed_image is not None and self.compare_mode:
            self.plot_histogram(compute_histogram(self.processed_image, self.hist_mode.currentText()), original=False)

    def reset_values(self):
        self.min_slider.setValue(0)
        self.max_slider.setValue(255)
        self.clip_limit_slider.setValue(30)
        
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processed_image = None
            self.last_applied = None
            self.compare_mode = False
            self.btn_compare.setText("Comparar Lado a Lado")
            self.show_image(self.current_image, original=True)
            self.plot_histogram(compute_histogram(self.current_image, self.hist_mode.currentText()), original=True)
            self.clear_processed_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec())