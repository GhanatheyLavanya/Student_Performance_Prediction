# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tkinter import *
import tkinter as tk
import tkinter.font as tkFont
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.messagebox
from tkhtmlview import HTMLLabel

        
def open_file():
        """Open a file for editing."""
        filepath = askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return
            txt_edit.delete(1.0, tk.END)
        with open(filepath, "r") as input_file:
            text = input_file.read()
            txt_edit.insert(tk.END, text) 
                
def save_file():
        """Save the current file as a new file."""
        filepath = asksaveasfilename(
        defaultextension="csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        with open(filepath, "w") as output_file:
            text = txt_edit.get(1.0, tk.END)
            output_file.write(text)
            tkinter.messagebox.showinfo('About File','Congratulations!!File Saved Successfully...')
            font_l= tkFont.Font(family="Times", size=27,weight=tkFont.BOLD)
            html_label=HTMLLabel(master=frame2,
            html='<a href="C:/Users/SUPPU SMILEY/Desktop/demo2/pred.html"><p style=" text-align: center">click here to see Predictions</p> </a>',
            fg="Black",
            bg="gray70",
            font=font_l)
            html_label.pack()
            txt_edit.destroy()
            label_c.destroy()
            btn_open.destroy()
            btn_save.destroy()

            

window = tk.Tk()
frame1 = tk.Frame(master=window,bg="gray17", bd=30 )
frame1.pack()
frame2 = tk.Frame(master=window, width=1400, height=600, bg="gray70")
frame2.pack()

fontStyle = tkFont.Font(family="Times", size=27,weight=tkFont.BOLD)
label1 = tk.Label(
    master=frame1,
    text="STUDENT PERFORMANCE PREDICTION USING ML",
    fg="white",
   bg="gray17",
    height=2,
    width=70,
    font=fontStyle
)
label1.pack()
fontStyle1 = tkFont.Font(family="Times", size=20,weight=tkFont.BOLD)
label2 = tk.Label(
    master=frame2,
    text="WELCOME TO STUDENT PERFORMANCE PREDICTION USING ML PROJECT!!!\n\n",
    fg="Black",
   bg="gray70",
    font=fontStyle1
)
fontStyle2 = tkFont.Font(family="Times", size=15,weight=tkFont.BOLD)
label3 = tk.Label(
    master=frame2,
    text="To get the predictions select and save your dataset at desired location. ",
    fg="Black",
   bg="gray70",
   font=fontStyle2
)

font1=tkFont.Font(family="Times",weight=tkFont.BOLD,size=15,underline=1)
label_c=tk.Label(master=frame2,text="Selected Dataset:",font=font1,fg="Black",bg="gray70")
font2=tkFont.Font(family="Times",weight=tkFont.BOLD)
txt_edit = tk.Text(window,width=70,height=15)

fr_buttons = tk.Frame(frame2, relief=tk.RAISED, bd=0,bg="gray70")
btn_open = tk.Button(fr_buttons, text="Open",font=font2, command=open_file)
btn_save = tk.Button(fr_buttons, text="Save As...",font =font2, command=save_file)
btn_open.pack(side=LEFT,padx=10)
btn_save.pack(side=LEFT,padx=10)
label_c.pack()
label_c.place(relx = 0.35, rely = 0.35, anchor = CENTER)
txt_edit.pack()
txt_edit.place(relx = 0.5, rely = 0.7, anchor = CENTER)
fr_buttons.pack(side=BOTTOM)
fr_buttons.place(relx = 0.5, rely = 0.9, anchor = CENTER)
label2.pack()
label2.place(relx = 0.5, rely = 0.1, anchor = N)
label3.pack()
label3.place(relx = 0.5, rely = 0.2, anchor = N)
window.mainloop()