from threading import Thread
import tkinter as tk
from DatasetHandler.dataset import Dataset
from IncrementalModel.model import Model
from tkinter import filedialog as fd
from PIL import Image,ImageTk

class MainForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.init_window()
        self.init_components()

    def init_components(self):
        self.container = tk.Frame(self, bg="#72b0a7")
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        buttonTrain = tk.Button(self.container, text="Train",
                                background="#ffffff", command=self.train_data)
        buttonTrain.pack(ipadx=10, ipady=10)
        buttonTrain.place(x=20, y=40)

        buttonTest = tk.Button(self.container, text="Test",
                               background="#ffffff", command=self.test_data)
        buttonTest.pack(ipadx=10, ipady=10)
        buttonTest.place(x=80, y=40)

    def init_window(self):
        self.title("Incremental Model App")
        self.geometry("800x600")

    def start(self):
        self.mainloop()

    def train_data(self):
        model = Model()
        dataset = Dataset()

        model.train(dataset.x_train, dataset.y_train,
                    dataset.x_test, dataset.y_test)
        model.save_model(1)

    def test_data(self):
        model = Model()
        model.load_model(1)
        filename = fd.askopenfilename()
        image = Image.open(filename)
        image = image.resize((300, 300), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas = tk.Canvas(self.container, width=300, height=300)
        self.canvas.pack()
        self.canvas.delete("all")
        self.canvas.create_image((0, 0), anchor=tk.NW, image=self.img)
        model.predict(filename)
