"""single_image_looper: Allows for repeated use of the
StructuralGT single image analysis while remembering the
settings from the previous analysis.

Copyright (C) 2021, The Regents of the University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov
Contact email: vecdrew@umich.edu
"""
from __main__ import *

from tkinter import *
from tkinter import filedialog

import settings
import cv2
import os
from PIL import Image, ImageTk


# This is almost identical to the regular single image selector, same UI
# Instead of calling make_settings from settings.py, it calls adjust_settings so the window is saved
# This makes running multiple images with similar settings but not as a batch much easier

def callback_click(event):

    # recording starting coordinates of click and drag crop
    entry_x1.delete(0, END)
    entry_y1.delete(0, END)
    entry_x1.insert('end', event.x)
    entry_y1.insert('end', event.y)


def callback_release(event):

    # recording ending coordinates of click and drag crop then calling to display the crop
    entry_x2.delete(0, END)
    entry_y2.delete(0, END)
    entry_x2.insert('end', event.x)
    entry_y2.insert('end', event.y)
    draw_rect()


def draw_rect():

    global crop_rect

    # remove previous crop data
    list = root.slaves()
    for l in list:
        if isinstance(l, Canvas):
            self = l
            try:
                self.delete(crop_rect)
            except NameError:
                None

    # getting width, height, and crop coordinates as ints
    w = self.winfo_width()
    h = self.winfo_height()
    cx1 = int(entry_x1.get())
    cx2 = int(entry_x2.get())
    cy1 = int(entry_y1.get())
    cy2 = int(entry_y2.get())

    # edge cases to make sure the crop is within the size of the image
    # (no negatives, smaller than total width, etc)
    # this also writes the coordinates in the entry boxes
    if (cx1 > w):
        cx1 = w - 1
        entry_x1.delete(0, END)
        entry_x1.insert('end', cx1)
    if (cx2 > w):
        cx2 = w
        entry_x2.delete(0, END)
        entry_x2.insert('end', cx2)
    if (cx1 < 0):
        cx1 = 0
        entry_x1.delete(0, END)
        entry_x1.insert('end', cx1)
    # you have to draw the rectangle from left to right
    if (cx2 < cx1):
        cx2 = cx1 + 1
        entry_x2.delete(0, END)
        entry_x2.insert('end', cx2)
    if (cy1 > h):
        cy1 = h - 1
        entry_y1.delete(0, END)
        entry_y1.insert('end', cy1)
    if (cy2 > h):
        cy2 = h
        entry_y2.delete(0, END)
        entry_y2.insert('end', cy2)
    if (cy1 < 0):
        cy1 = 0
        entry_y1.delete(0, END)
        entry_y1.insert('end', cy1)
    # you also have to draw the rectangle from top to bottom
    if (cy2 < cy1):
        cy2 = cy1 + 1
        entry_y2.delete(0, END)
        entry_y2.insert('end', cy2)
    # making the new rectangle and displaying it with a red outline
    crop_rect = self.create_rectangle(cx1, cy1, cx2, cy2, outline='Red', width=3.0)


def getfile():

    # asking the user to select their image
    afile = filedialog.askopenfilename()

    # testing if file is a workable image
    if afile.endswith(('.tif', '.png', '.jpg')):

        #removing the previous canvas image
        list = root.slaves()
        for l in list:
            if isinstance(l, Canvas):
                l.destroy()

        # splitting the file location into the filename and path and displaying
        path, filename = os.path.split(afile)
        fileentry.config(state='normal')
        fileentry.delete('1.0', END)
        fileentry.insert('end', afile)
        fileentry.config(font=("Helvetica", 12, "italic"))
        fileentry.config(state='disabled')

        # automatically configuring the saving directory to the same folder as the image
        savedir.config(state='normal')
        savedir.delete('1.0', END)
        savedir.insert('end', path)
        savedir.config(font=("Helvetica", 12, "italic"))
        savedir.config(state='disabled')

    # file has to be an image that we can work with
    else:
        fileentry.config(state='normal')
        fileentry.delete('1.0', END)
        fileentry.insert('end', 'File needs to be a .tif, .png, or .jpg')
        fileentry.config(font=("Helvetica", 14, "bold"))
        fileentry.config(state='disabled')


def choosesave():

    # asking for where they want to save the file and displaying the directory in the box
    saveloc = filedialog.askdirectory()
    savedir.config(state='normal')
    savedir.delete('1.0', END)
    savedir.insert('end', saveloc)
    savedir.config(font=("Helvetica", 12, "italic"))
    savedir.config(state='disabled')


def Confirm_button():

    # getting the filename and save location to send to settings.py
    path, filename = os.path.split(fileentry.get("1.0", END))
    filename = filename.replace("\n", "")
    saveloc = savedir.get("1.0", END)
    saveloc = saveloc.replace("\n", "")

    # redrawing the rectangle with the most recent coordinates given
    draw_rect()
    cx1 = int(entry_x1.get())
    cx2 = int(entry_x2.get())
    cy1 = int(entry_y1.get())
    cy2 = int(entry_y2.get())

    # standardizing image size and redrawing it so it matches the crop coordinates
    raw_src = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
    w, h = raw_src.shape
    if (h > w):
        scalefactor = 512 / h
    else:
        scalefactor = 512 / w
    swidth = int(scalefactor * w)
    sheight = int(scalefactor * h)
    dsize = (sheight, swidth)
    src1 = cv2.resize(raw_src, dsize)

    # cropping the image, then sending everything through to settings.py
    src = src1[cy1:cy2, cx1:cx2]
    return src, saveloc, filename


def Confirm_button_ex():

    # sending the cropped image
    src, saveloc, filename = Confirm_button()
    settings.adjust_settings(root, src, saveloc, filename)


def Confirm_nocrop_button():

    # getting the file name and directory to send through to settings.py
    path, filename = os.path.split(fileentry.get("1.0", END))
    filename = filename.replace("\n", "")
    saveloc = savedir.get("1.0", END)
    saveloc = saveloc.replace("\n", "")

    # standardizing image size, then resizing it and sending it through to settings.py
    raw_src = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
    w, h = raw_src.shape
    if (h > w):
        scalefactor = 512 / h
    else:
        scalefactor = 512 / w
    swidth = int(scalefactor * w)
    sheight = int(scalefactor * h)
    dsize = (sheight, swidth)
    src = cv2.resize(raw_src, dsize)
    return src, saveloc, filename


def Confirm_nocrop_button_ex():

    # send to settings.py without cropping the image
    src, saveloc, filename = Confirm_nocrop_button()
    settings.adjust_settings(root, src, saveloc, filename)


def newimwindow():

    # global A for image storage
    global A

    # getting the image location and opening it
    im_file = fileentry.get("1.0", END)
    path, filename = os.path.split(im_file)
    filename = filename.replace("\n", "")
    PIL_image = Image.open(os.path.join(path, filename))

    # getting the image size and resizing the image to standardize it
    w1, h1 = PIL_image.size
    if (h1 > w1):
        scalefactor = 512 / h1
    else:
        scalefactor = 512 / w1
    PIL_image = PIL_image.resize(size=(int(w1 * scalefactor), int(h1 * scalefactor)))

    # creating the global image object
    A = ImageTk.PhotoImage(image=PIL_image)
    h = A.height()
    w = A.width()

    # creating the canvas for the image, then creating the image anchored in the upper left
    canvas = Canvas(root, width=w, height=h)
    canvas.pack()
    canvas.create_image(0, 0, image=A, anchor=NW)

    # when you click on the canvas it attempts to build a crop rectangle
    canvas.bind("<ButtonPress-1>", callback_click)
    canvas.bind("<ButtonRelease-1>", callback_release)

    # if the user changes the crop with the text boxes this is to call to redraw the rectangle
    button6 = Button(frame4, text="Update", command=draw_rect)
    button6.grid(row=2, column=1)


def make_gui():

    # Making a window and declaring a few variables
    global root
    root = Tk()
    root.title("StructuralGT GUI")

    A = None
    src = []
    saveloc = str("")


    # creating the frames for the window
    global frame1,frame2,frame3,frame4

    frame1 = Frame(root)
    frame2 = Frame(root)
    frame3 = Frame(root)
    frame4 = Frame(root)

    frame1.pack()
    frame2.pack()
    frame3.pack(side=BOTTOM)
    frame4.pack(side=BOTTOM)

    # building the gui itself, this is all just the text labels
    label1 = Label(frame1, text="Please select a file\nOptionally, you can select a region of an image with Crop")
    filelabel = Label(frame1, text="File:")
    saveplace = Label(frame1, text="Result save directory:")
    labelx = Label(frame4, text="Crop x:", anchor=W)
    labely = Label(frame4, text="Crop y:", anchor=W)
    comma1 = Label(frame4, text=",")
    comma2 = Label(frame4, text=",")

    # the user cannot manually enter a file entry or save directory to avoid errors
    global fileentry,savedir

    fileentry = Text(frame1, height=2, bg="gray", state='disabled')
    savedir = Text(frame1, height=2, bg="gray", state='disabled')

    # crop coordinate values go in these text boxes
    global entry_x1,entry_x2,entry_y1,entry_y2

    entry_x1 = Entry(frame4)
    entry_x2 = Entry(frame4)
    entry_y1 = Entry(frame4)
    entry_y2 = Entry(frame4)

    # all of the buttons and calling their respective function calls
    #button0 = Button(frame3, text="Proceed with crop", bg="Green", command=Confirm_button_ex)
    button1 = Button(frame3, text="Proceed without crop", fg="Black", command=Confirm_nocrop_button_ex)
    button2 = Button(frame3, text="Exit", bg="Red", command=frame3.quit)
    button3 = Button(frame2, text="Select file...", bg="gray", command=getfile)
    button4 = Button(frame2, text="Choose Save Location...", bg="gray", command=choosesave)
    #button5 = Button(frame2, text="Crop Image", bg="gray", command=newimwindow)

    # setting the location of all elements in the gui
    label1.grid(row=0, column=1)
    filelabel.grid(row=1, column=0)
    saveplace.grid(row=2, column=0)
    labelx.grid(row=0, column=0)
    entry_x1.grid(row=0, column=1)
    comma1.grid(row=0, column=2)
    entry_x2.grid(row=0, column=3)
    labely.grid(row=1, column=0)
    entry_y1.grid(row=1, column=1)
    comma2.grid(row=1, column=2)
    entry_y2.grid(row=1, column=3)

    fileentry.grid(row=1, column=1)
    savedir.grid(row=2, column=1)

    #button0.grid(row=0, column=0)
    button1.grid(row=0, column=1)
    button2.grid(row=0, column=2)
    button3.grid(row=0, column=0)
    button4.grid(row=0, column=1)
    #button5.grid(row=0, column=2)

    # keep the window alive
    root.mainloop()
