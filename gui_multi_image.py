"""gui_multi_image: Generates the graphical user interface for
performing multi-image graph theory analysis on a folder of images.

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

import multi_image_settings
import cv2
import os


def getfolder():

    # asking the user to select their folder
    global foldername
    foldername = filedialog.askdirectory()
    folderentry.config(state='normal')
    folderentry.delete('1.0', END)
    folderentry.insert('end', foldername)
    folderentry.config(font=("Helvetica", 12, "italic"))
    folderentry.config(state='disabled')

    savedir.config(state='normal')
    savedir.delete('1.0', END)
    savedir.insert('end', foldername)
    savedir.config(font=("Helvetica", 12, "italic"))
    savedir.config(state='disabled')


def choosesave():

    # asking for where they want to save the file and displaying the directory in the box
    saveloc = filedialog.askdirectory()
    savedir.config(state='normal')
    savedir.delete('1.0', END)
    savedir.insert('end', saveloc)
    savedir.config(font=("Helvetica", 12, "italic"))
    savedir.config(state='disabled')


def Confirm_button():

    # getting the file names and directory to send through to settings.py
    files = os.listdir(foldername)
    files = sorted(files, key=str.lower)
    path = foldername
    images = []
    filenames = []
    # testing if file is a workable image
    for afile in files:
        if afile.endswith(('.tif', '.png', '.jpg', '.jpeg')):
            print(afile)
            filename = afile
            # splitting the file location into the filename and path and displaying
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
            images.append(src)
            filenames.append(filename)

        # file has to be an image that we can work with
        else:
            pass

    saveloc = savedir.get("1.0", END)
    saveloc = saveloc.replace("\n", "")

    # returning the list of
    return images, saveloc, filenames


def Confirm_button_ex():

    # send to settings.py without cropping the image
    images, saveloc, filenames = Confirm_button()

    multi_image_settings.make_settings(root, images, saveloc, filenames)


def make_gui(window, multi):
    multi.destroy()
    window.destroy()

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
    label1 = Label(frame1, text="Please select a folder\nOnly .tif .png and .jpg files will be included")
    filelabel = Label(frame1, text="Folder:")
    saveplace = Label(frame1, text="Result save directory:")

    # the user cannot manually enter a file entry or save directory to avoid errors
    global folderentry,savedir

    folderentry = Text(frame1, height=2, bg="gray", state='disabled')
    savedir = Text(frame1, height=2, bg="gray", state='disabled')

    # all of the buttons and calling their respective function calls
    button0 = Button(frame3, text="Proceed", bg="Green", command=Confirm_button_ex)
    button1 = Button(frame3, text="Exit", bg="Red", command=frame3.quit)
    button2 = Button(frame2, text="Select folder...", bg="gray", command=getfolder)
    button3 = Button(frame2, text="Choose Save Location...", bg="gray", command=choosesave)

    # setting the location of all elements in the gui
    label1.grid(row=0, column=1)
    filelabel.grid(row=1, column=0)
    saveplace.grid(row=2, column=0)

    folderentry.grid(row=1, column=1)
    savedir.grid(row=2, column=1)

    button0.grid(row=0, column=0)
    button1.grid(row=0, column=2)
    button2.grid(row=0, column=0)
    button3.grid(row=0, column=1)

    # keep the window alive
    root.mainloop()
