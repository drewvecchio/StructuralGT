"""StructuralGT: A python package for automated graph theory analysis of structural networks.
Designed for processing digital micrographs of complex network materials.
For example, analyzing SEM images of polymer network.

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

from tkinter import *
import tkinter.messagebox

from StructuralGT import gui_multi_image, gui_single_image

window = Tk()
window.title("StructuralGT Image Entry Selection")

# All this really does is let you choose single or multi image options

def about_window():
    about_info = "StructuralGT: Automated graph theory analysis of network materials. " \
                 "Performs image detection and graph theory calculation on digital images of networks." \
                 "\nCopyright (C) 2021, The Regents of the University of Michigan" \
                 "\n\n" \
                 "This program is free software: you can redistribute it and/or modify " \
                 "it under the terms of the GNU General Public License as published by " \
                 "the Free Software Foundation, either version 3 of the License, or " \
                 "(at your option) any later version." \
                 "\n\n" \
                 "This program is distributed in the hope that it will be useful, " \
                 "but WITHOUT ANY WARRANTY; without even the implied warranty of " \
                 "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the " \
                 "GNU General Public License for more details." \
                 "\n\n" \
                 "You should have received a copy of the GNU General Public License " \
                 "along with this program.  If not, see <https://www.gnu.org/licenses/>."

    tkinter.messagebox.showinfo('About StructuralGT', about_info)

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
button_about = Button(frame_2, text="About...", command=about_window)
label1 = Label(frame_1, text="Please select an Image entry method")

label1.grid(row=1, column=0)
button1.grid(row=2, column=0)
button2.grid(row=2, column=1)
button_about.grid(row=3, column=0)

window.mainloop()
