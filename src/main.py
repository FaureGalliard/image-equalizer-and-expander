import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QHBoxLayout,
    QSlider, QFormLayout, QMessageBox, QGridLayout
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
        self.setWindowTitle("Procesador de Histogramas Avanzado")
        self.setGeometry(100, 100, 1200, 800)
        self.initialize_variables()
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controles superiores
        self.setup_top_controls(main_layout)

        # Ajustes del histograma
        self.setup_histogram_settings(main_layout)

        # Área de visualización
        self.display_widget = QWidget()
        self.display_layout = QGridLayout(self.display_widget)
        main_layout.addWidget(self.display_widget, stretch=1)

        # Initialize display area
        self.setup_display_area()

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
        self.clip_limit_slider.setRange(0, 50)
        self.clip_limit_slider.setValue(0)
        self.clip_limit_slider.setTickInterval(1)
        self.clip_limit_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        settings_layout.addRow("Mínimo (Expansión):", self.min_slider)
        settings_layout.addRow("Máximo (Expansión):", self.max_slider)
        settings_layout.addRow("Límite Clip (Ecualización):", self.clip_limit_slider)

        main_layout.addLayout(settings_layout)

    def setup_display_area(self):
        # Clear existing layout
        while self.display_layout.count():
            item = self.display_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create widgets
        self.image_label_original = QLabel()
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")

        self.image_label_processed = QLabel()
        self.image_label_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_processed.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")

        self.canvas_original = FigureCanvas(Figure(figsize=(5, 3), dpi=100))
        self.canvas_original.setStyleSheet("background-color: white;")

        self.canvas_processed = FigureCanvas(Figure(figsize=(5, 3), dpi=100))
        self.canvas_processed.setStyleSheet("background-color: white;")

        if self.compare_mode:
            # Four-container layout for comparison mode (side by side)
            self.display_layout.addWidget(QLabel("Imagen Original"), 0, 0)
            self.display_layout.addWidget(QLabel("Histograma Original"), 0, 1)
            self.display_layout.addWidget(QLabel("Imagen Procesada"), 0, 2)
            self.display_layout.addWidget(QLabel("Histograma Procesado"), 0, 3)
            self.display_layout.addWidget(self.image_label_original, 1, 0)
            self.display_layout.addWidget(self.canvas_original, 1, 1)
            self.display_layout.addWidget(self.image_label_processed, 1, 2)
            self.display_layout.addWidget(self.canvas_processed, 1, 3)
            self.display_layout.setRowStretch(1, 2)
        else:
            # Two-container layout for normal mode
            self.display_layout.addWidget(QLabel("Imagen Actual"), 0, 0)
            self.display_layout.addWidget(QLabel("Histograma Actual"), 0, 1)
            self.display_layout.addWidget(self.image_label_original, 1, 0)
            self.display_layout.addWidget(self.canvas_original, 1, 1)
            self.display_layout.setRowStretch(1, 2)

    def setup_connections(self):
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_reset.clicked.connect(self.reset_values)
        self.btn_compare.clicked.connect(self.toggle_comparison)
        self.min_slider.valueChanged.connect(self.update_custom_expansion)
        self.max_slider.valueChanged.connect(self.update_custom_expansion)
        self.clip_limit_slider.valueChanged.connect(self.update_custom_equalization)

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
                self.setup_display_area()
                self.show_image(self.current_image, original=True)
                self.plot_histogram(compute_histogram(self.current_image), original=True)
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
            
            if not isinstance(histogram, list):
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
        
        self.setup_display_area()
        
        if self.compare_mode:
            self.show_image(self.original_image, original=True)
            self.plot_histogram(compute_histogram(self.original_image), original=True)
            if self.processed_image is not None:
                self.show_image(self.processed_image, original=False)
                self.plot_histogram(compute_histogram(self.processed_image), original=False)
            else:
                self.clear_processed_display()
        else:
            self.show_image(self.current_image, original=True)
            self.plot_histogram(compute_histogram(self.current_image), original=True)
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
                self.plot_histogram(compute_histogram(self.processed_image), original=False)
            else:
                self.show_image(self.processed_image, original=True)
                self.plot_histogram(compute_histogram(self.processed_image), original=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar expansión: {e}")

    def update_custom_equalization(self):
        if self.current_image is None:
            return
        try:
            self.processed_image = equalize_histogram_custom(
                self.current_image,
                self.clip_limit_slider.value() / 10000.0
            )
            self.last_applied = "equalization"
            if self.compare_mode:
                self.show_image(self.processed_image, original=False)
                self.plot_histogram(compute_histogram(self.processed_image), original=False)
            else:
                self.show_image(self.processed_image, original=True)
                self.plot_histogram(compute_histogram(self.processed_image), original=True)
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
            self.setup_display_area()
            self.show_image(self.current_image, original=True)
            self.plot_histogram(compute_histogram(self.current_image), original=True)
            self.clear_processed_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec())