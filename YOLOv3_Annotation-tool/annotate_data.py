from __future__ import division
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import os
import glob
import random
import config

# colors for the bboxes
COLORS = ['red', 'blue', 'olive', 'teal', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256


class AnnotationTool():
	def __init__(self, master):
		# set up the main frame
		self.parent = master
		self.parent.title("AnnotationTool")
		self.frame = Frame(self.parent)
		self.frame.pack(fill=BOTH, expand=1)

		self.parent.resizable(width=FALSE, height=FALSE)

		# initialize global state
		self.imageDir          = ''
		self.imageList         = []
		self.egDir             = ''
		self.egList            = []
		self.outDir            = ''
		self.cur               = 0
		self.total             = 0
		self.category          = -1
		self.imagename         = ''
		self.labelfilename     = ''
		self.tkimg             = None
		self.currentLabelclass = ''
		self.classNames        = []
		self.classContainer    = config.PARAMS['classNames']
		self.imageFormat       = config.PARAMS['imageFormat']
		self.sourceImageDir    = config.PARAMS['imageDirectory']
		self.annotationDir     = config.PARAMS['annotationDirectory']

		# initialize mouse state
		self.STATE = {}
		self.STATE['click'] = 0
		self.STATE['x'], self.STATE['y'] = 0, 0

		# reference to bbox
		self.bboxIdList = []
		self.bboxId = None
		self.bboxList = []
		self.hl = None
		self.vl = None

		# ----------------- GUI stuff ---------------------
		# dir entry & load
		self.label = Label(self.frame, text="Image Dir:")
		self.label.grid(row=0, column=0, sticky=E)
		self.entry = Entry(self.frame)
		self.entry.grid(row=0, column=1, sticky=W + E)
		self.ldBtn = Button(self.frame, text="Load", command=self.loadDir)
		self.ldBtn.grid(row=0, column=2, sticky=W + E)

		# main panel for labeling
		self.mainPanel = Canvas(self.frame, cursor='tcross')
		self.mainPanel.bind("<Button-1>", self.mouseClick)
		self.mainPanel.bind("<Motion>", self.mouseMove)
		# press <Espace> to cancel current bbox
		self.parent.bind("<Escape>", self.cancelBBox)
		self.parent.bind("s", self.cancelBBox)
		self.parent.bind("a", self.prevImage)  # press 'a' to go backforward
		self.parent.bind("d", self.nextImage)  # press 'd' to go forward
		self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

		# choose class
		self.lb1 = Label(self.frame, text='Choose Class')
		self.lb1.grid(row=3, column=3, padx=5, sticky=N + W)

		self.classname = StringVar()
		self.classcandidate = ttk.Combobox(
			self.frame, state='readonly', textvariable=self.classname)
		self.classcandidate.grid(row=4, column=3, padx=5, sticky=N)

		for className in self.classContainer.split():
			self.classNames.append(className)

		# print self.classNames
		if self.classNames:
			self.classcandidate['values'] = self.classNames
		else:
			self.classcandidate['values'] = ['null']
		self.classcandidate.current(0)

		# add class
		self.classEntry = Entry(self.frame)
		self.classEntry.grid(row=5, column=3, padx=5, sticky=W + E)
		self.addBtn = Button(self.frame, text="Add Class",
							 command=self.addClass)

		self.addBtn.grid(row=6, column=3, padx=5, sticky=W + E)

		# showing bbox info & delete bbox
		self.lb2 = Label(self.frame, text='Bounding boxes:')
		self.lb2.grid(row=3, column=2,  sticky=W + N)
		self.listbox = Listbox(self.frame, width=22, height=12)
		self.listbox.grid(row=4, column=2, sticky=N + S)
		self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
		self.btnDel.grid(row=5, column=2, sticky=W + E + N)
		self.btnClear = Button(
			self.frame, text='ClearAll', command=self.clearBBox)
		self.btnClear.grid(row=6, column=2, sticky=W + E + N)

		# control panel for image navigation
		self.ctrPanel = Frame(self.frame)
		self.ctrPanel.grid(row=7, column=1, columnspan=2, sticky=W + E)
		self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
		self.prevBtn.pack(side=LEFT, padx=5, pady=3)
		self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
		self.nextBtn.pack(side=LEFT, padx=5, pady=3)
		self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
		self.progLabel.pack(side=LEFT, padx=5)
		self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
		self.tmpLabel.pack(side=LEFT, padx=5)
		self.idxEntry = Entry(self.ctrPanel, width=5)
		self.idxEntry.pack(side=LEFT)
		self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
		self.goBtn.pack(side=LEFT)

		# display mouse position
		self.disp = Label(self.ctrPanel, text='')
		self.disp.pack(side=RIGHT)

		self.frame.columnconfigure(1, weight=1)
		self.frame.rowconfigure(4, weight=1)

	def addClass(self):
		className = self.classEntry.get()
		print("add class : ", className)
		if className:
			self.classNames.append(className)
		if self.classNames:
			self.classcandidate['values'] = self.classNames
		else:
			self.classcandidate['values'] = ['null']
		self.classcandidate.current(0)

	def loadDir(self, dbg=False):
		if not dbg:
			s = self.entry.get()
			self.parent.focus()
			self.category = s

		# self.imageDir = os.path.join(self.sourceImageDir, '%03d' % (self.category))
		self.imageDir = os.path.join(self.sourceImageDir, s)

		if not os.path.exists(self.imageDir):
			print("No such directory present : ", self.imageDir)
			return

		for format in self.imageFormat:
			for file in glob.glob(os.path.join(self.imageDir, ('*.' + format))):
				self.imageList.append(file)

		if len(self.imageList) == 0:
			print('No images found in the specified dir!')
			return

		# default to the 1st image in the collection
		self.cur = 1
		self.total = len(self.imageList)

		# set up output dir
		self.outDir = os.path.join(self.annotationDir, s)
		if not os.path.exists(self.outDir):
			os.mkdir(self.outDir)

		self.loadImage()
		print ('%d images loaded from %s' % (self.total, s))

	def loadImage(self):
		# load image
		imagepath = self.imageList[self.cur - 1]
		self.img = Image.open(imagepath)
		self.tkimg = ImageTk.PhotoImage(self.img)
		self.mainPanel.config(width=max(self.tkimg.width(), 400),
							  height=max(self.tkimg.height(), 400))
		self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
		self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

		# load labels
		self.clearBBox()
		self.imagename = os.path.split(imagepath)[-1].split('.')[0]
		labelname = self.imagename + '.txt'
		self.labelfilename = os.path.join(self.outDir, labelname)
		bbox_cnt = 0
		if os.path.exists(self.labelfilename):
			with open(self.labelfilename) as f:
				for (i, line) in enumerate(f):

					elems = line.split(' ')
					print("Line : ", line, " \n")
					c = elems[0]
					x_norm = float(elems[1])
					y_norm = float(elems[2])
					w_norm = float(elems[3])
					h_norm = float(elems[4])

					img_width = self.img.size[0]
					img_height = self.img.size[1]

					x_max = int((x_norm * img_width) + ((w_norm * img_width) / 2))
					x_min = int(x_max - (w_norm * img_width))
					y_max = int((y_norm * img_height) + ((h_norm * img_height) / 2))
					y_min = int(y_max - (h_norm * img_height))

					tmp = (str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + c).split()
					temp = [c , str(x_norm), str(y_norm), str(w_norm), str(h_norm)]

					self.bboxList.append(tuple(temp))
					tmpId = self.mainPanel.create_rectangle(int(tmp[0]), int(tmp[1]),
															int(tmp[2]), int(
																tmp[3]),
															width=2,
															outline=COLORS[(len(self.bboxList) - 1) % len(COLORS)])

					self.bboxIdList.append(tmpId)
					self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' % (tmp[4], int(tmp[0]), int(tmp[1]),
																			int(tmp[2]), int(tmp[3])))
					self.listbox.itemconfig(
						len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

	def saveImage(self):
		if self.labelfilename:
			with open(self.labelfilename, 'w') as f:
				for bbox in self.bboxList:
					f.write(' '.join(map(str, bbox)) + '\n')
			print ('Image No. %d saved' % (self.cur))

	def mouseClick(self, event):
		if self.STATE['click'] == 0:
			self.STATE['x'], self.STATE['y'] = event.x, event.y
		else:
			self.currentLabelclass = self.getClass()
			x1, x2 = min(self.STATE['x'], event.x), max(
				self.STATE['x'], event.x)
			y1, y2 = min(self.STATE['y'], event.y), max(
				self.STATE['y'], event.y)

			dw = 1. / self.img.size[0]
			dh = 1. / self.img.size[1]
			x = (x1 + x2) / 2.0
			y = (y1 + y2) / 2.0
			w = x2 - x1
			h = y2 - y1
			x = x * dw
			y = y * dh
			w = w * dw
			h = h * dh

			self.bboxList.append((self.currentLabelclass, x, y, w, h))

			self.bboxIdList.append(self.bboxId)
			self.bboxId = None
			self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' %
								(self.currentLabelclass, x1, y1, x2, y2))
			self.listbox.itemconfig(
				len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
		self.STATE['click'] = 1 - self.STATE['click']

	def mouseMove(self, event):
		self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
		if self.tkimg:
			if self.hl:
				self.mainPanel.delete(self.hl)
			self.hl = self.mainPanel.create_line(
				0, event.y, self.tkimg.width(), event.y, width=2)
			if self.vl:
				self.mainPanel.delete(self.vl)
			self.vl = self.mainPanel.create_line(
				event.x, 0, event.x, self.tkimg.height(), width=2)
		if 1 == self.STATE['click']:
			if self.bboxId:
				self.mainPanel.delete(self.bboxId)
			self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'],
														  event.x, event.y,
														  width=2,
														  outline=COLORS[len(self.bboxList) % len(COLORS)])

	def cancelBBox(self, event):
		if 1 == self.STATE['click']:
			if self.bboxId:
				self.mainPanel.delete(self.bboxId)
				self.bboxId = None
				self.STATE['click'] = 0

	def delBBox(self):
		sel = self.listbox.curselection()
		if len(sel) != 1:
			return
		idx = int(sel[0])
		self.mainPanel.delete(self.bboxIdList[idx])
		self.bboxIdList.pop(idx)
		self.bboxList.pop(idx)
		self.listbox.delete(idx)

	def clearBBox(self):
		for idx in range(len(self.bboxIdList)):
			self.mainPanel.delete(self.bboxIdList[idx])
		self.listbox.delete(0, len(self.bboxList))
		self.bboxIdList = []
		self.bboxList = []

	def prevImage(self, event=None):
		self.saveImage()
		if self.cur > 1:
			self.cur -= 1
			self.loadImage()

	def nextImage(self, event=None):
		self.saveImage()
		if self.cur < self.total:
			self.cur += 1
			self.loadImage()

	def gotoImage(self):
		idx = int(self.idxEntry.get())
		if 1 <= idx and idx <= self.total:
			self.saveImage()
			self.cur = idx
			self.loadImage()

	def getClass(self):
		return self.classcandidate.get()


if __name__ == '__main__':
	root = Tk()
	tool = AnnotationTool(root)
	root.resizable(width=True, height=True)
	root.mainloop()
