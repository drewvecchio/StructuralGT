"""StructuralGT: A python package for automated graph theory analysis of structural networks.
Designed for processing digital micrographs of complex network materials.
For example, analyzing SEM images of polymer network.
Copyright (C) 2021, The Regents of the University of Michigan.
Author:drewvecchio
"""

from tkinter import *

import gui_single_image
import gui_multi_image

window = Tk()
window.title("StructuralGT Image Entry Selection")

# All this really does is let you choose single or multi image options

def multi_image():

    # This just gives a small warning before you select multi image settings
    # there isn't one for single image settings because that should be the default choice
    multi = Tk()
    multi.title("Multi-Image Entry")
    mframe_1 = Frame(multi)
    mframe_2 = Frame(multi)
    mframe_1.pack()
    mframe_2.pack()

    label_multi = Label(mframe_1, text="This method is for analysis of multiple images at the same time\n"
                                        "All images must be separate files with proper file extensions\n"
                                        "Note: You will NOT be able to crop and preview images before analysis\n"
                                        "Please use the single-image option to test settings and preview images "
                                        "before proceeding")

    button_back = Button(mframe_2, text="Go Back", command=multi.destroy)
    button_proceed_multi = Button(mframe_2, text="Proceed", command=lambda: gui_multi_image.make_gui(window, multi))

    label_multi.grid(row=0, column=0)
    button_proceed_multi.grid(row=0, column=1)
    button_back.grid(row=0, column=0)

    multi.mainloop()


frame_1 = Frame(window)
frame_2 = Frame(window)

frame_1.pack()
frame_2.pack()

button1 = Button(frame_2, text="Single Image StructuralGT", bg="Green", command=lambda: gui_single_image.make_gui(window))
button2 = Button(frame_2, text="Multi-Image StructuralGT", bg="Light Blue", command=multi_image)
label1 = Label(frame_1, text="Please select an Image entry method")

label1.grid(row=1, column=0)
button1.grid(row=2, column=0)
button2.grid(row=2, column=1)

window.mainloop()
