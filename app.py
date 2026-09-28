from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import ( 
                             QApplication, QWidget, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QCheckBox, QMainWindow, QScrollArea, QVBoxLayout,
                             QSizePolicy, QDialog, QDialogButtonBox, QMessageBox
                             )
    
from PyQt6.QtGui import QImage, QPixmap

from math import dist
import cv2    
import random
import numpy as np
import matplotlib.pyplot as plt

MOVE = "Переместить"
ADD = "Добавить/Удалить"
LINK = "Связать"
EDIT = "Изменить"

RADIUS = 20

TABLE_TEXT = "(0-0.2]: Несущественное воздействие\n(0.2-0.4]: Малое воздействие\n(0.4-0.6]: Среднее воздействие\n(0.6-0.8]: Умеренное воздействие\n(0.8-1.0]: Сильное воздействие".split("\n")
class Neuron:
    def __init__(self, short_name: str, long_name: str, x: int, y: int) -> None:
        self.short_name = short_name
        self.long_name = long_name
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f"{self.short_name} {self.long_name}"

class AddNeuronWeight(QDialog):
    def __init__(self, parent = None) -> None:
        super().__init__(parent)
            
        self.setWindowTitle("Добавить связь")
        self.setModal(True)
            
        self.weight = QLineEdit()
        self.weight.setPlaceholderText("Коэффициент связи")
    
        buttons = (
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel 
        )
            
        buttonBox = QDialogButtonBox(buttons)
            
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
            
        layout = QVBoxLayout()
        layout.addWidget(self.weight)
        layout.addWidget(buttonBox)
            
        self.setLayout(layout)
            
    def get_weight(self):
        return float(self.weight.text()) 
        
    def accept(self) -> None:
        try:
            if -1 <= float(self.weight.text()) <= 1:
                return super().accept()
            else:
                QMessageBox.warning(self, "Ошибка", "Коэффициент должен быть в [-1; 1]")  
            return
        except:
            QMessageBox.warning(self, "Ошибка", "Введен неправильный коэффициент")
            return
        
        
class AddNeuronName(QDialog):
    def __init__(self, parent = None) -> None:
        super().__init__(parent)
        
        self.setWindowTitle("Добавить нейрон")
        self.setModal(True)
        
        self.name = QLineEdit()
        self.name.setPlaceholderText("Название нейрона")

        buttons = (
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel 
        )
        
        buttonBox = QDialogButtonBox(buttons)
        
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.name)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def get_name(self):
        return self.name.text() 
    
    def accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "Ошибка", "Введено пустое имя нейрона")
            return
        return super().accept()
    
class StartEvaluation(QDialog):
    def __init__(self, matrix, neuronList: list[Neuron], parent = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Запустить симуляцию")
        self.setModal(True)
        
        self.checkboxes: list[QCheckBox] = []
        self.editLines: list[QLineEdit] = []
        scrollArea = QScrollArea()
        self.neuronList = neuronList
        size = len(neuronList)
        self.matrix = matrix[:size, :size]
        neuronBox_layout = QVBoxLayout()
        
        for neuron in self.neuronList:
            neuron_layout = QHBoxLayout()
            checkbox = QCheckBox()
            editLine = QLineEdit()
            editLine.setText("0")
            self.editLines.append(editLine)
            self.checkboxes.append(checkbox)
            
            neuron_layout.addWidget(checkbox)
            neuron_layout.addWidget(QLabel(f"{neuron.short_name} {neuron.long_name}"))
            neuron_layout.addWidget(editLine)
            
            neuron_container = QWidget()
            neuron_container.setLayout(neuron_layout)
            neuronBox_layout.addWidget(neuron_container)
            
        neuronBox_container = QWidget()
        neuronBox_container.setLayout(neuronBox_layout)
        scrollArea.setWidget(neuronBox_container)
        
        self.cycles = QLineEdit()
        
        buttons = (
                    QDialogButtonBox.StandardButton.Ok
                    | QDialogButtonBox.StandardButton.Cancel 
                )
        buttonBox = QDialogButtonBox(buttons)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scrollArea)
        main_layout.addWidget(self.cycles)
        main_layout.addWidget(buttonBox)
        
        self.setLayout(main_layout)
    
    def procced(self):
        size = len(self.neuronList)
        times = int(self.cycles.text())+1
        values = np.zeros((times, size))
        values_to_print = np.array([checkbox.isChecked() for checkbox in self.checkboxes])
        names = [self.neuronList[i].short_name for i in range(len(self.neuronList)) if values_to_print[i]]
        p = np.array([int(editLine.text()) if str.isdigit(editLine.text()) else 0 for editLine in self.editLines ])
        values[1] = p
        p = p @ self.matrix
        for time in range(2, times):
            values[time] = values[time-1] + p @ self.matrix
            p = p @ self.matrix
        values = values.transpose()[values_to_print]
        
        
        for index in range(len(names)):
            plt.plot(values[index], label = names[index])
        plt.legend()
        plt.show()
        
            
    def accept(self) -> None:
        if not str.isdigit(self.cycles.text()) or int(self.cycles.text()) < 1:
            return
        return super().accept()
        
class EditNeuron(QDialog):
    def __init__(self, name, parent = None) -> None:
        super().__init__(parent)
        
        self.setWindowTitle("Изменить нейрон")
        self.setModal(True)
        
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("Название нейрона")

        buttons = (
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel 
        )
        
        buttonBox = QDialogButtonBox(buttons)
        
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(self.name)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def get_name(self):
        return self.name.text() 
    
    def accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "Ошибка", "Введено пустое имя нейрона")
            return
        return super().accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.status = False
        self.count = 1
        self.neuronList: list[Neuron] = []
        self.capacity = 10
        self.links = np.zeros((self.capacity, self.capacity))
        self.image_width, self.image_height = 1000, 1000
        self.image = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)
        self.current_mode = ''
        self.hovered_neuron = -1
        self.selected_neuron = -1
        self.GUI_init()
    
    def reallocate(self):
        new_links = np.zeros((self.capacity*2, self.capacity*2))
        new_links[:self.capacity, :self.capacity] = self.links
        self.capacity *= 2
        self.links = new_links
    
    def delete_neuron(self, index):
        self.links = np.delete(self.links, index, axis=0)
        self.links = np.delete(self.links, index, axis=1)
        
        self.links = np.insert(self.links, self.capacity-1, np.zeros(self.capacity-1), axis = 1)
        self.links = np.insert(self.links, self.capacity-1, np.zeros(self.capacity), axis = 0)
    
    def GUI_init(self):
        self.set_global_style()
        
        add_button = QPushButton(ADD)
        add_button.clicked.connect(lambda: self.set_current_choise(ADD))
        
        edit_button = QPushButton(EDIT)
        edit_button.clicked.connect(lambda: self.set_current_choise(EDIT))
        
        connect_button = QPushButton(LINK)
        connect_button.clicked.connect(lambda: self.set_current_choise(LINK))
        
        move_button = QPushButton(MOVE)
        move_button.clicked.connect(lambda: self.set_current_choise(MOVE))
        
        choise_layout = QVBoxLayout()
        self.choise_buttons_list: list[QPushButton] = [move_button, add_button, connect_button, edit_button]
        
        for button in self.choise_buttons_list:
            choise_layout.addWidget(button)
        
        choise_layout.addStretch()
        choise_container = QWidget()
        choise_container.setLayout(choise_layout)
        
        self.label_image = QLabel()
        self.label_image.installEventFilter(self)
        self.label_image.setMaximumSize(1000, 1000)
        self.label_image.setMouseTracking(True)
        self.showImage()
        
        self.info_scrollArea = QScrollArea()
        self.info_layout = QVBoxLayout()
        self.info_container =  QWidget()
        self.update_info()
        self.info_container.setLayout(self.info_layout)
        
        self.info_scrollArea.setFixedSize(300, 500)
        self.info_scrollArea.setWidget(self.info_container)
        
        self.check_label = QLabel("Ошибка: Нет вершин")
        
        table_layout = QVBoxLayout()
        for text in TABLE_TEXT:
            label = QLabel(text)
            table_layout.addWidget(label)
        
        table_container = QWidget()
        table_container.setLayout(table_layout)
        
        start_button = QPushButton("Начать")
        start_button.clicked.connect(self.start)
        
        
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.info_scrollArea)
        status_layout.addWidget(self.check_label)
        status_layout.addWidget(table_container)
        status_layout.addWidget(start_button)
        status_layout.addStretch()
        
        status_container = QWidget()
        status_container.setLayout(status_layout)
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(choise_container)
        main_layout.addWidget(self.label_image)
        main_layout.addWidget(status_container)
        
        main = QWidget()
        main.setLayout(main_layout)
        
        self.setCentralWidget(main)
        
    def start(self):
        if self.status:
            dlg = StartEvaluation(self.links, self.neuronList)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                dlg.procced()
            
    
    def update_info(self):
        new_info_container = QWidget()
        old = self.info_scrollArea.takeWidget()
        if old:
            old.setParent(None)
            old.deleteLater()
        
        
        new_info_layout = QVBoxLayout()
        for i in range(len(self.neuronList)):
            neuron = self.neuronList[i]
            neuron.short_name = f"x{i+1}"
            neuron_info_layout = QHBoxLayout()
            label_short = QLabel(neuron.short_name)
            label_long = QLabel(neuron.long_name)
            label_long.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            neuron_info_layout.addWidget(label_short)
            neuron_info_layout.addWidget(label_long)
            
            neuron_info = QWidget()
            neuron_info.setLayout(neuron_info_layout)
            
            new_info_layout.addWidget(neuron_info)
        new_info_container.setLayout(new_info_layout)
        self.info_scrollArea.setWidget(new_info_container)

    def showImage(self):
        height, width, _ = self.image.shape
        bytes_per_line = 3 * width
        pixmap = QImage(self.image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888) #type: ignore
        pixmap = QPixmap.fromImage(pixmap)
        self.label_image.setPixmap(pixmap)
        
    def set_current_choise(self, mode: str):
        self.selected_neuron = -1
        if (self.current_mode == mode):
            self.current_mode = ''
        else:
            self.current_mode = mode
        for button in self.choise_buttons_list:
            button.setProperty("active", False)
            if (button.text() == self.current_mode):
                button.setProperty("active", True)
            button.style().unpolish(button)  #type: ignore
            button.style().polish(button)    #type: ignore
        
    def set_global_style(self):
            # Единый CSS для всех кнопок
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-weight: bold;
                    border: 2px solid #2980b9;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                
                QPushButton[active="true"] {
                    background-color: #27da23;
                    border: 2px solid #1a8a16;
                }
                QPushButton[active="true"]:hover {
                    background-color: #2ecc71;
                }
            """)    
    
    def check_status(self):
        size = len(self.neuronList)
        available_neurons = self.links[:size, :size]
        #print(available_neurons)
        self.status = False
        if any(np.count_nonzero(available_neurons, axis=0) == 0): 
            self.check_label.setText("Как минимум у одной вершины нет входных связей")
        elif any(np.count_nonzero(available_neurons, axis=1) == 0): 
            self.check_label.setText("Как минимум у одной вершины нет выходных связей")
        else:
            self.check_label.setText("Все хорошо")
            self.status = True
            
    
    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:
        if a0 != self.label_image: return super().eventFilter(a0, a1)
        if a1.type() == QEvent.Type.MouseButtonPress and self.current_mode == ADD:
            mouse_event = a1
            if mouse_event.button() == Qt.MouseButton.RightButton: # type: ignore
                pos = mouse_event.pos()  # Или mouse_event.pos() # type: ignore
                x = pos.x()
                y = pos.y()
                #print(f"Правая кнопка нажата на label в координатах: ({x:.1f}, {y:.1f})")
                index = self.find_at(x,y)
                if index != -1:
                    self.neuronList.pop(index)
                    self.delete_neuron(index)
                    self.update_info()
                    self.check_status()
                    self.redraw_image()

                return True
            elif mouse_event.button() == Qt.MouseButton.LeftButton: # type: ignore       
                pos = mouse_event.pos()  # Или mouse_event.pos() # type: ignore
                x = pos.x()
                y = pos.y()
                #print(f"Левая кнопка нажата на label в координатах: ({x:.1f}, {y:.1f})")
                
                dlg = AddNeuronName(self)
                
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    name = dlg.get_name()
                    neuron = Neuron(f"x{len(self.neuronList)+1}", name, x, y)
                    self.count+=1
                    self.neuronList.append(neuron)          
                    if self.capacity < len(self.neuronList):
                        self.reallocate()
                    self.draw_neuron(x, y, text = neuron.short_name)
                    self.update_info()
                    self.check_status()
                    self.showImage()
                return True
        elif a1.type() == QEvent.Type.MouseButtonPress and a1.button() == Qt.MouseButton.LeftButton and self.current_mode == MOVE: #type: ignore
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            index = self.find_at(x, y)
            if (index != -1):
                self.selected_neuron = index
            return True
        elif a1.type() == QEvent.Type.MouseButtonRelease and a1.button() == Qt.MouseButton.LeftButton and self.current_mode == MOVE: #type: ignore
            self.selected_neuron = -1
        elif a1.type() == QEvent.Type.MouseMove and self.current_mode == MOVE:
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            if (self.selected_neuron == -1):
                self.hovered_neuron = self.find_at(x, y)
                self.redraw_image()
            else:
                neuron = self.neuronList[self.selected_neuron]
                neuron.x = int(x)
                neuron.y = int(y)
                self.redraw_image()
            return True
        elif self.current_mode == LINK and a1.type() == QEvent.Type.MouseButtonPress and a1.button() == Qt.MouseButton.LeftButton: #type: ignore
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            index = self.find_at(x, y)
            if (index != -1):
                self.selected_neuron = index
            return True
        elif self.current_mode == LINK and a1.type() == QEvent.Type.MouseMove:
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            if (self.selected_neuron == -1):
                self.hovered_neuron = self.find_at(x, y)
                self.redraw_image()
            else:
                self.hovered_neuron = self.find_at(x,y)
                self.redraw_image()
                neuron = self.neuronList[self.selected_neuron]
                cv2.arrowedLine(self.image, (neuron.x, neuron.y), (int(x), int(y)), (0, 0, 0), 1)
                self.showImage()
        elif self.current_mode == LINK and a1.type() == QEvent.Type.MouseButtonRelease:
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            index = self.find_at(x, y)
            #print(self.links)
            if (index != -1 and index != self.selected_neuron and self.selected_neuron != -1):
                if (self.links[self.selected_neuron, index] == 0 and self.links[index, self.selected_neuron] == 0):
                    dlg = AddNeuronWeight(self)                
                    if dlg.exec() == QDialog.DialogCode.Accepted:
                        weight = dlg.get_weight()
                        self.links[self.selected_neuron, index] = weight
                else:
                    self.links[self.selected_neuron, index] = 0
                    self.links[index, self.selected_neuron] = 0
                self.check_status()
                self.redraw_image()
            self.selected_neuron = -1
        elif self.current_mode == EDIT and a1.type() == QEvent.Type.MouseButtonPress and a1.button() == Qt.MouseButton.LeftButton: #type: ignore
            pos = a1.position() #type: ignore
            x = pos.x()
            y = pos.y()
            index = self.find_at(x, y)
            if index != -1:
                neuron = self.neuronList[index]
                dlg = EditNeuron(neuron.long_name)
                
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    new_name = dlg.get_name()
                    neuron.long_name = new_name
                    self.update_info()
                    return True
        return super().eventFilter(a0, a1)
    
    def find_at(self, x, y):
        index = -1
        for i, neuron in enumerate(self.neuronList):
            neuron = self.neuronList[i]
            if dist((neuron.x, neuron.y), (x,y)) < 0.8 * RADIUS: #type: ignore
                index = i
                break
        return index
    
    def redraw_image(self):
        self.image = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)
        for i in range(len(self.neuronList)):
            neuron = self.neuronList[i]
            if (i == self.hovered_neuron or i == self.selected_neuron):
                self.draw_neuron(neuron.x, neuron.y, color = (0, 0, 255), text = neuron.short_name)
            else:
                self.draw_neuron(neuron.x, neuron.y, text = neuron.short_name)      
            for index in range(self.capacity):
                if self.links[i][index] != 0:
                    neuron_to = self.neuronList[index]
                    val = self.links[i][index]
                    self.draw_line(neuron.x, neuron.y, neuron_to.x, neuron_to.y, val, (0, 0, 255) if val > 0 else (255, 0, 0))
                
        self.showImage()
            
    
    def draw_neuron(self, x, y, color = (0, 0, 0), text = ""):
        cv2.circle(self.image, (int(x), int(y)), RADIUS, color, 1)
        cv2.putText(self.image, text, (int(x) - 4*len(text), int(y) + 6), 1, 1, (0, 0, 0))
    
    def draw_line(self, x1, y1, x2, y2, val, color = (0, 0, 255)):
        off_x, off_y = self.calculate_offset(x1, y1, x2, y2)
        if (x1 > x2): off_x *= -1
        if (y1 < y2): off_y *= -1
        cv2.arrowedLine(self.image, (x1 + off_x, y1 - off_y), (x2 - off_x, y2 + off_y), color, 2, tipLength=0.1)
        cv2.putText(self.image, f"{val:.2f}", ((x1 + x2)//2, (y1 + y2)//2), 1, 1, (0, 0, 0))
    
    def calculate_offset(self, x1, y1, x2, y2):     
        if (x2 != x1): 
            dx = (y2 - y1) / (x2 - x1)
        else:
            dx = float("INF")
        angle = np.arctan(dx)
        return (int(RADIUS * np.cos(angle)), int(RADIUS * abs(np.sin(angle))))
        
app = QApplication([])

window = MainWindow()
window.show()

app.exec()