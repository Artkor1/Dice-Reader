from PyQt5 import QtCore, QtGui, QtWidgets
import os
import re
import cv2
import traceback
import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms
from detecto import core, utils, visualize

from qt_material import apply_stylesheet
#https://github.com/UN-GCPDS/qt-material#install
#pip install qt-material


# worker thread signals
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)


# worker thread
class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()



class Ui_MainWindow(object):

    #variables
    main_directory = ""
    model = ""
    image = []
    image_path = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 970)
        MainWindow.setMinimumSize(QtCore.QSize(1300, 970))
        MainWindow.setMaximumSize(QtCore.QSize(1300, 970))
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_image = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_image.setGeometry(QtCore.QRect(370, 170, 911, 601))
        self.groupBox_image.setTitle("")
        self.groupBox_image.setObjectName("groupBox_image")
        self.label_image = QtWidgets.QLabel(self.groupBox_image)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 891, 581))
        self.label_image.setText("")
        self.label_image.setScaledContents(True)
        self.label_image.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_image.setObjectName("label_image")
        self.pushButton_load_images = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_load_images.setGeometry(QtCore.QRect(110, 100, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_load_images.setFont(font)
        self.pushButton_load_images.setObjectName("pushButton_load_images")
        self.listWidget_images = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_images.setGeometry(QtCore.QRect(30, 170, 321, 731))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.listWidget_images.setFont(font)
        self.listWidget_images.setObjectName("listWidget_images")
        self.groupBox_load_images = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_load_images.setGeometry(QtCore.QRect(20, 10, 461, 71))
        self.groupBox_load_images.setTitle("")
        self.groupBox_load_images.setObjectName("groupBox_load_images")
        self.label_folder = QtWidgets.QLabel(self.groupBox_load_images)
        self.label_folder.setGeometry(QtCore.QRect(10, 10, 71, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_folder.setFont(font)
        self.label_folder.setObjectName("label_folder")
        self.pushButton_predict = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_predict.setGeometry(QtCore.QRect(880, 100, 261, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_predict.setFont(font)
        self.pushButton_predict.setObjectName("pushButton_predict")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1160, 10, 121, 101))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.checkBox_labels = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_labels.setGeometry(QtCore.QRect(10, 10, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_labels.setFont(font)
        self.checkBox_labels.setChecked(True)
        self.checkBox_labels.setObjectName("checkBox_labels")
        self.checkBox_scores = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_scores.setGeometry(QtCore.QRect(10, 40, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_scores.setFont(font)
        self.checkBox_scores.setChecked(True)
        self.checkBox_scores.setObjectName("checkBox_scores")
        self.checkBox_boxes = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_boxes.setGeometry(QtCore.QRect(10, 70, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_boxes.setFont(font)
        self.checkBox_boxes.setChecked(True)
        self.checkBox_boxes.setObjectName("checkBox_boxes")
        self.groupBox_results = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_results.setGeometry(QtCore.QRect(370, 780, 321, 131))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_results.setFont(font)
        self.groupBox_results.setTitle("")
        self.groupBox_results.setObjectName("groupBox_results")
        self.label_total = QtWidgets.QLabel(self.groupBox_results)
        self.label_total.setGeometry(QtCore.QRect(160, 10, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_total.setFont(font)
        self.label_total.setObjectName("label_total")
        self.label_average_score = QtWidgets.QLabel(self.groupBox_results)
        self.label_average_score.setGeometry(QtCore.QRect(10, 90, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_average_score.setFont(font)
        self.label_average_score.setObjectName("label_average_score")
        self.label_results = QtWidgets.QLabel(self.groupBox_results)
        self.label_results.setGeometry(QtCore.QRect(10, 10, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_results.setFont(font)
        self.label_results.setObjectName("label_results")
        self.label_average_dice = QtWidgets.QLabel(self.groupBox_results)
        self.label_average_dice.setGeometry(QtCore.QRect(10, 50, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_average_dice.setFont(font)
        self.label_average_dice.setObjectName("label_average_dice")
        self.groupBox_save = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_save.setGeometry(QtCore.QRect(710, 780, 571, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_save.setFont(font)
        self.groupBox_save.setTitle("")
        self.groupBox_save.setObjectName("groupBox_save")
        self.label_save = QtWidgets.QLabel(self.groupBox_save)
        self.label_save.setGeometry(QtCore.QRect(10, 10, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_save.setFont(font)
        self.label_save.setObjectName("label_save")
        self.lineEdit_save = QtWidgets.QLineEdit(self.groupBox_save)
        self.lineEdit_save.setGeometry(QtCore.QRect(130, 10, 431, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_save.setFont(font)
        self.lineEdit_save.setObjectName("lineEdit_save")
        self.groupBox_type = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_type.setGeometry(QtCore.QRect(500, 10, 361, 71))
        self.groupBox_type.setTitle("")
        self.groupBox_type.setObjectName("groupBox_type")
        self.comboBox_type = QtWidgets.QComboBox(self.groupBox_type)
        self.comboBox_type.setGeometry(QtCore.QRect(110, 10, 241, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox_type.setFont(font)
        self.comboBox_type.setObjectName("comboBox_type")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.label_type = QtWidgets.QLabel(self.groupBox_type)
        self.label_type.setGeometry(QtCore.QRect(10, 20, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_type.setFont(font)
        self.label_type.setObjectName("label_type")
        self.groupBox_model = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_model.setGeometry(QtCore.QRect(500, 90, 361, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_model.setFont(font)
        self.groupBox_model.setTitle("")
        self.groupBox_model.setObjectName("groupBox_model")
        self.label_model = QtWidgets.QLabel(self.groupBox_model)
        self.label_model.setGeometry(QtCore.QRect(10, 10, 61, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_model.setFont(font)
        self.label_model.setObjectName("label_model")
        self.comboBox_model = QtWidgets.QComboBox(self.groupBox_model)
        self.comboBox_model.setGeometry(QtCore.QRect(80, 10, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox_model.setFont(font)
        self.comboBox_model.setObjectName("comboBox_model")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.groupBox_threshold = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_threshold.setGeometry(QtCore.QRect(870, 10, 281, 71))
        self.groupBox_threshold.setTitle("")
        self.groupBox_threshold.setObjectName("groupBox_threshold")
        self.SpinBox_threshold = QtWidgets.QDoubleSpinBox(self.groupBox_threshold)
        self.SpinBox_threshold.setGeometry(QtCore.QRect(130, 10, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.SpinBox_threshold.setFont(font)
        self.SpinBox_threshold.setDecimals(2)
        self.SpinBox_threshold.setMaximum(100.0)
        self.SpinBox_threshold.setProperty("value", 0.8)
        self.SpinBox_threshold.setObjectName("SpinBox_threshold")
        self.label = QtWidgets.QLabel(self.groupBox_threshold)
        self.label.setGeometry(QtCore.QRect(10, 10, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(890, 860, 231, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_save.setFont(font)
        self.pushButton_save.setObjectName("pushButton_save")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 29))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dice Reader"))
        self.pushButton_load_images.setText(_translate("MainWindow", "Load images"))
        self.label_folder.setText(_translate("MainWindow", "Folder:"))
        self.pushButton_predict.setText(_translate("MainWindow", "Predict results"))
        self.checkBox_labels.setText(_translate("MainWindow", "Labels"))
        self.checkBox_scores.setText(_translate("MainWindow", "Scores"))
        self.checkBox_boxes.setText(_translate("MainWindow", "Boxes"))
        self.label_total.setText(_translate("MainWindow", "Total sum:"))
        self.label_average_score.setText(_translate("MainWindow", "Average score:"))
        self.label_results.setText(_translate("MainWindow", "Results:"))
        self.label_average_dice.setText(_translate("MainWindow", "Average dice:"))
        self.label_save.setText(_translate("MainWindow", "Save image:"))
        self.lineEdit_save.setText(_translate("MainWindow", "my_image.png"))
        self.comboBox_type.setItemText(0, _translate("MainWindow", "D6 dots"))
        self.comboBox_type.setItemText(1, _translate("MainWindow", "D6 D8 D10 D12"))
        self.label_type.setText(_translate("MainWindow", "Dice type:"))
        self.label_model.setText(_translate("MainWindow", "Model:"))
        self.comboBox_model.setItemText(0, _translate("MainWindow", "Faster R-CNN transforms"))
        self.comboBox_model.setItemText(1, _translate("MainWindow", "Faster R-CNN"))
        self.label.setText(_translate("MainWindow", "Threshold:"))
        self.pushButton_save.setText(_translate("MainWindow", "Save image"))

    def setupUi2(self, MainWindow):
        self.lineEdit_images_folder = DropLineEdit(self.groupBox_load_images)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_images_folder.setFont(font)
        self.lineEdit_images_folder.setGeometry(QtCore.QRect(81, 10, 340, 51))
        self.lineEdit_images_folder.setObjectName("lineEdit_images_folder")
        self.lineEdit_images_folder.setText(os.getcwd())

        self.pushButton_save.setMinimumHeight(51)
        self.pushButton_load_images.setFixedHeight(51)
        self.pushButton_predict.setFixedHeight(51)
        self.pushButton_save.setFixedHeight(51)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_load_images.clicked.connect(self.button_load_images)
        self.listWidget_images.clicked.connect(self.button_show_image)
        self.comboBox_type.currentTextChanged.connect(self.load_model)
        self.comboBox_model.currentTextChanged.connect(self.load_model)
        self.pushButton_predict.clicked.connect(self.button_predict)
        self.pushButton_save.clicked.connect(self.save_image)

        self.threadpool = QtCore.QThreadPool()
        self.load_model()

    def button_load_images(self):
        if self.threadpool.activeThreadCount() > 0:
            self.gui_show_message("Please wait")
            return 0
        image_files = []
        self.main_directory = self.lineEdit_images_folder.text()
        if os.path.exists(self.main_directory) and os.path.isdir(self.main_directory):
            files = os.listdir(self.main_directory)
        else:
            self.gui_show_message("Incorrect folder")
            return 0

        file_matcher = r'*(.png|.jpg|.jpeg)'
        for file in files:
            if re.match(r'.*(.png|.jpg|.jpeg)', file):
                image_files.append(file)
        self.listWidget_images.clear()
        for i in range(len(image_files)):
            self.listWidget_images.insertItem(i, image_files[i])

    def button_show_image(self):
        if self.threadpool.activeThreadCount() > 0:
            self.gui_show_message("Please wait")
            return 0
        self.reset_summary()
        item = self.listWidget_images.currentItem()
        image_path = self.main_directory + "\\" + item.text()
        if os.path.exists(image_path):
            self.gui_show_message("Selected " + image_path)
            self.label_image.setPixmap(QtGui.QPixmap(image_path))
            self.image = cv2.imread(image_path)
            self.image_path = image_path
        else:
            self.gui_show_message("Incorrect image file")
            return 0

    def load_model(self):
        try:
            dice_type = self.comboBox_type.currentText()
            model_version = self.comboBox_model.currentText()
            path = ""
            model = ""

            if dice_type == "D6 dots":
                if model_version == "Faster R-CNN transforms":
                    path = "models/model_d6_dots_transforms.pth"
                elif model_version == "Faster R-CNN":
                    path = "models/model_d6_dots.pth"
            elif dice_type == "D6 D8 D10 D12":
                if model_version == "Faster R-CNN transforms":
                    path = "models/model_d6d8d10d12_transforms.pth"
                elif model_version == "Faster R-CNN":
                    path = "models/model_d6d8d10d12.pth"

            if os.path.exists(path):
                if dice_type == "D6 dots":
                    self.model = core.Model.load(path, ['one', 'two', 'three', 'four', 'five', 'six'])
                elif dice_type == "D6 D8 D10 D12":
                    self.model = core.Model.load(path, ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'])
            else:
                self.gui_show_message("Missing model")
                self.model = ""
                return 0
            message = "Model: " + dice_type + ", " + model_version + " loaded"
            self.gui_show_message(message)
        except:
            self.gui_show_message("Error loading model")

    def button_predict(self):
        if self.threadpool.activeThreadCount() > 0:
            self.gui_show_message("Please wait")
            return 0
        worker = Worker(self.thread_predict)
        self.threadpool.start(worker)

    def thread_predict(self):
        self.reset_summary()
        if not self.model:
            self.gui_show_message("Missing model")
            return 0
        if not len(self.image):
            self.gui_show_message("Missing image")
            return 0
        self.gui_show_message("Predicting image in progress")
        self.image = cv2.imread(self.image_path)
        predictions = self.model.predict(self.image)
        labels, boxes, scores = predictions
        boxes = boxes.tolist()
        scores = scores.tolist()
        for i in range(len(scores)):
            scores[i] = round(scores[i], 4)
        labels, boxes, scores = self.remove_below_threshold(labels, boxes, scores)
        self.summary(labels, scores)

        if self.checkBox_scores.isChecked():
            labels = self.add_scores(labels, scores)

        if not len(labels) == len(boxes) == len(scores):
            self.gui_show_message("Error in data")
            return 0

        if not labels or not boxes or not scores:
            self.gui_show_message("No results found, try lowering threshold")
            return 0

        if self.checkBox_boxes.isChecked():
            try:
                self.insert_boxes(boxes)
            except:
                self.gui_show_message("Error marking boxes on image")
                return 0

        if self.checkBox_labels.isChecked():
            try:
                self.insert_labels(labels, boxes)
            except:
                self.gui_show_message("Error marking labels on image")
                return 0

        cv2.imwrite("temp_image.png", self.image)
        self.label_image.setPixmap(QtGui.QPixmap("temp_image.png"))
        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")
        else:
            self.gui_show_message("No temp_image.png file found")
        self.gui_show_message("Finished predicting")

    def reset_summary(self):
        self.label_results.setText("Results: ")
        self.label_total.setText("Total sum: ")
        self.label_average_dice.setText("Average dice: ")
        self.label_average_score.setText("Average score: ")

    def summary(self, labels, scores):
        results = len(labels)
        self.label_results.setText("Results: " + str(results))

        total = 0
        labels_int = numbers_to_int(labels)
        for label_int in labels_int:
            total += label_int
        self.label_total.setText("Total sum: " + str(total))

        if results != 0:
            average_dice = round(total/results, 2)
        else:
            average_dice = 0
        self.label_average_dice.setText("Average dice: " + str(average_dice))

        if len(scores) != 0:
            average_score = round(sum(scores) / len(scores), 2)
        else:
            average_score = 0
        self.label_average_score.setText("Average score: " + str(average_score))
        return 0

    def insert_boxes(self, boxes):
        color = (0, 0, 255)
        image_height = self.image.shape[0]
        image_width = self.image.shape[1]
        thickness = int(min(image_height, image_width) / 400)
        if thickness < 1:
            thickness = 1
        for i in range(len(boxes)):
            start_point = (int(boxes[i][0]), int(boxes[i][1]))
            end_point = (int(boxes[i][2]), int(boxes[i][3]))
            image = cv2.rectangle(self.image, start_point, end_point, color, thickness)

    def insert_labels(self,labels, boxes):
        font = cv2.FONT_ITALIC
        image_height = self.image.shape[0]
        image_width = self.image.shape[1]
        font_scale = min(image_height, image_width) / 600
        color = (0, 0, 255)
        thickness = int(min(image_height, image_width) / 300)
        if thickness < 1:
            thickness = 1

        for i in range(len(labels)):
            start_point = (int(boxes[i][0]), int(boxes[i][1]))
            text = labels[i]
            image = cv2.putText(self.image, text, start_point, font, font_scale, color, thickness)

    def add_scores(self, labels, scores):
        for i in range(len(labels)):
            labels[i] = labels[i] + ": " + str(scores[i])
        return labels

    def remove_below_threshold(self, labels, boxes, scores):
        threshold = self.SpinBox_threshold.value()
        new_labels = []
        new_boxes = []
        new_scores = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                new_scores.append(scores[i])
                new_labels.append(labels[i])
                new_boxes.append(boxes[i])
        return new_labels, new_boxes, new_scores

    def save_image(self):
        if self.threadpool.activeThreadCount() > 0:
            self.gui_show_message("Please wait")
            return 0
        if not len(self.image):
            self.gui_show_message("Missing image")
            return 0
        self.gui_show_message("Saving image")
        image_name = self.lineEdit_save.text()
        #image_pattern = r'.*(.png|.jpg|.jpeg)'
        #image_pattern = r'\.(png|jpg|jpeg)$'
        #if re.match(image_pattern, image_name, flags=re.IGNORECASE):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            cv2.imwrite(image_name, self.image)
            self.gui_show_message("Image saved")
        else:
            self.gui_show_message("Incorrect name, use png, jpg or jpeg")

    def gui_show_message(self, message):
        print(message)
        self.statusbar.showMessage(message)
        app.processEvents()

def numbers_to_int(numbers_list):
    int_list = []
    numbers_dict = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    for number in numbers_list:
        int_list.append(numbers_dict[number])
    return int_list

# modified version of QLineEdit class, that allows files/folders and web links to be drag and dropped
class DropLineEdit(QtWidgets.QLineEdit):
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        md = event.mimeData()

        if md.hasUrls():
            files = []
            for url in md.urls():
                files.append(url.toLocalFile())
            self.setText(" ".join(files))
            event.acceptProposedAction()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_lightgreen.xml')

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.setupUi2(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
