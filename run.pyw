#!/usr/bin/env python
try:
	import game
except Exception:
	def print_error(msg: str):
		crashfile = "crashes/crash_" + str(int(time.time())) + ".txt"
		try:
			log = open(crashfile, "w")
		except FileNotFoundError:
			os.mkdir("crashes")
			msg += "Warning: The crash log directory was missing and had to be created!\n"
			log = open(crashfile, "w")
		print(msg)
		log.write(msg)
		log.close()
	try:
		import os, traceback, time
		import sys
		ex = traceback.format_exc()
		message = ""
		crashfile = "crashes/crash_" + str(int(time.time())) + ".txt"
		message += "Your game has crashed due to an unexpected error.\nPython has raised the following exception:\n" + \
		           "".join(ex) + "\nThe exception has been logged in the file \n" + crashfile + " ."
		try:
			from OpenGL.GL import *
			from OpenGL.GLU import *
			import pygame as pg
			from pygame.locals import *
		except ImportError:
			message += "\nCouldn't load necessary libraries!"
			if sys.platform == "win32":
				os.system("start criticalcrash.bat 0 \"" +
						message.replace('"', "'").replace("\n", '" "') + '"')
			print_error(message)
			quit()
		try:
			pg.display.set_icon(pg.image.load("icon.ico"))
		except Exception:
			message += "\nWarning: The game icon seems to be missing!\nCheck if there's a file named 'icon.ico' in the game directory and try to replace it with a placeholder if missing!\n"
		try:
			from init import UI, load_textures
		except Exception:
			message += "\nError: Your game files seem to be corrupted!"
			if sys.platform == "win32":
				os.system("start criticalcrash.bat 2 \"" +
						message.replace('"', "'").replace("\n", '" "') + '"')
			print_error(message)
			quit()
		pg.mouse.set_visible(True)

		if sys.platform == "win32":
			ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ESCrashWindow")

		pg.display.set_caption("Uh-Oh: Your game crashed!")

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glDisable(GL_TEXTURE_2D)
		glUseProgram(0)
		glClearColor(0.31, 0.39, 0.71, 1.0)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		glBegin(GL_QUADS)
		glVertex2f(-0.7, -0.7)
		glVertex2f(-0.7, 0.7)
		glVertex2f(0.7, 0.7)
		glVertex2f(0.7, -0.7)
		glEnd()

		try:
			ASCIILoc = load_textures("unicodeL.png")
			TableHeight = 24
		except (pg.error, FileNotFoundError):
			try:
				ASCIILoc = load_textures("ascii.png")
				TableHeight = 16
			except (pg.error, FileNotFoundError):
				try:
					ASCIILoc = load_textures("../default/unicodeL.png")
					TableHeight = 24
				except (pg.error, FileNotFoundError):
					ASCIILoc = load_textures("../default/ascii.png")
					TableHeight = 16

		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, ASCIILoc[0])
		UI.write(message, (-0.6, 0.6), 0.05, chardim=(0.65, 1.0), color=(0, 0, 0))
		pg.display.flip()
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		print_error(message)
		while True:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					quit()
	except Exception as e:
		message += "\nUnexpected error during crash handling! \n" + "".join(e)
		if platform == "win32":
			os.system("start criticalcrash.bat 1 \"" +
					message.replace('"', "'").replace("\n", '" "') + '"')
		print_error(message)
		quit()