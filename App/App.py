from ast import Mod
from threading import Thread
import tkinter as tk

import matplotlib
from numpy import shape
from DatasetHandler.dataset import Dataset
from IncrementalModel.model import Model
from tkinter import Label, filedialog as fd
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import numpy as np
from tkinter import font as tkFont


class MainForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.init_window()
        self.init_components()
        self.init_model()

    def init_components(self):
        self.container = tk.Frame(self, bg="#72b0a7")
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        helv16 = tkFont.Font(family='Helvetica', size=16, weight=tkFont.BOLD)

        buttonTrain = tk.Button(self.container, text="Train",
                                background="#ffffff", command=self.train_data, font=helv16)
        buttonTrain.pack(ipadx=30, ipady=30)
        buttonTrain.place(x=20, y=40)

        buttonTest = tk.Button(self.container, text="Test",
                               background="#ffffff", command=self.test_data, font=helv16)
        buttonTest.pack(ipadx=30, ipady=30)
        buttonTest.place(x=120, y=40)

    def init_window(self):
        self.title("Incremental Model App")
        self.geometry("1024x720")

    def start(self):
        self.mainloop()

    def init_model(self):
        self.model = Model()
        self.dataset = Dataset()

    def train_data(self):
        #items_per_task = [5000,5000,100,100,100,100,100,100,100,100] uncomment lines for separate tasks
        #nr_examples = len(self.dataset.x_train)
        items_per_task = 1000
        max_task = 10
        
        ratio = 0.1
        
        #label = 0
        for i in range(0, max_task):
            #task_nr = i / items_per_task + 1
            print(f"Task {i+1}/{max_task}")
            # x_task:list
            # y_task:list
            
            # if i>4:
            #     (x_task, y_task) = self.dataset.get_data_by_label(items_per_task[i-5], 0, label)
            #     label+=1
            # else:
            #     (x_task, y_task) = self.dataset.get_data(1000,i*1000)
            (x_task, y_task) = self.dataset.get_data(items_per_task,i*items_per_task)
            if i != 0:
                # if i>4:
                #     self.dataset.max_number_of_rand_images = items_per_task[i-5]*10;
                # else:
                #     self.dataset.max_number_of_rand_images = 1000;
                self.dataset.max_number_of_rand_images = int(items_per_task*ratio);
                self.dataset.geneate_random_images()
                (x_generated, y_generated) = self.model.evaluate_generated_dataset(
                    self.dataset.max_number_of_rand_images)
                x_task = np.concatenate((x_task, x_generated))
                y_task = np.concatenate((y_task, y_generated))
            
            self.model.train(x_task, y_task,
                             self.dataset.x_test, self.dataset.y_test)



        matplotlib.use('TkAgg')

        accuracy_containter = tk.Frame(
            self.container, bg="#ffffff", height=500, width=500)
        accuracy_containter.pack(side="top", fill="both", expand=True)
        accuracy_containter.place(x=20, y=150)
        accuracy_containter.grid_columnconfigure(0, weight=1)
        accuracy_containter.grid_rowconfigure(0, weight=1)

        figure_accuracy = Figure(figsize=(4, 4), dpi=100)
        accuracy = figure_accuracy.add_subplot()
        accuracy.set_title("Accuracy", loc='left')
        line1, = accuracy.plot(
            self.model.custom_stats_callback.model_train_accuracy, linewidth=0.75)
        line2, = accuracy.plot(
            self.model.custom_stats_callback.model_value_accuracy, linewidth=0.75)
        figure_accuracy.legend(
            (line1, line2), ('Train Accuracy', 'Value Accuracy'), 'upper right')

        figure_accuracy_canvas = FigureCanvasTkAgg(
            figure_accuracy, accuracy_containter)
        NavigationToolbar2Tk(figure_accuracy_canvas, accuracy_containter)
        figure_accuracy_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        loss_containter = tk.Frame(
            self.container, bg="#ffffff", height=500, width=500)
        loss_containter.pack(side="top", fill="both", expand=True)
        loss_containter.place(x=520, y=150)
        loss_containter.grid_columnconfigure(0, weight=1)
        loss_containter.grid_rowconfigure(0, weight=1)

        figure_loss = Figure(figsize=(4, 4), dpi=100)
        loss = figure_loss.add_subplot()
        loss.set_title("Loss", loc='left')
        loss_line1, = loss.plot(
            self.model.custom_stats_callback.model_train_loss, linewidth=0.75)
        loss_line2, = loss.plot(
            self.model.custom_stats_callback.model_value_loss, linewidth=0.75)
        figure_loss.legend(
            (loss_line1, loss_line2), ('Train Loss', 'Value Loss'), 'upper right')

        figure_loss_canvas = FigureCanvasTkAgg(figure_loss, loss_containter)
        NavigationToolbar2Tk(figure_loss_canvas, loss_containter)
        figure_loss_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.model.save_model(4)

    def test_data(self):
        self.model.load_model(4)
        filename = fd.askopenfilename()
        image = Image.open(filename)
        image = image.resize((300, 300), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas = tk.Canvas(self.container, width=300, height=300)
        self.canvas.pack()
        self.canvas.delete("all")
        self.canvas.create_image((0, 0), anchor=tk.NW, image=self.img)
        prediction = self.model.predict(filename)

        label = Label(master=self.container,
                      text=f"Prediction: {prediction}")
        label.place(x=400,
                    y=320)
