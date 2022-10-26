#!/usr/bin/env python
try:
	import game
except Exception:
	try:
		import os, traceback, time
		import sys
		ex = traceback.format_exc()
		print(ex)
		message = ""
		time = "crashes/crash_" + str(int(time.time())) + ".txt"
		try:
			log = open(time, "w")
		except FileNotFoundError:
			os.mkdir("crashes")
			message = message + "Warning: The crash log directory was missing and had to be created!\n"
			log = open(time, "w")
		log.write("".join(ex))
		log.close()

		message = message + "Your game has crashed due to an unexpected error.\nPython has raised the following exception:\n" + "".join(
			ex) + "\nThe exception has been logged in the file \n" + time + " ."
		
		try:
			import OpenGL_accelerate
		except ImportError:
			pass
		try:
			from OpenGL.GL import *
			from OpenGL.GLU import *
			import pygame as pg
			from pygame.locals import *
		except ImportError:
			if sys.platform == "win32":
				os.system("start criticalcrash.bat 0 \"" +
						message.replace('"', "'").replace("\n", '" "') + '"')
			quit()
		try:
			pg.display.set_icon(pg.image.load("icon.ico"))
		except Exception:
			message = message + "\nWarning: The game icon seems to be missing!\nCheck if there's a file named 'icon.ico' in the game directory and try to replace it with a placeholder if missing!\n"
		try:
			from init import UI, LoadTextures, Unicode
		except Exception:
			if sys.platform == "win32":
				os.system("start criticalcrash.bat 2 \"" +
						message.replace('"', "'").replace("\n", '" "') + '"')
			quit()
		pg.mouse.set_visible(True)

		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ESCrashWindow")
		pg.display.set_caption("Uh-Oh: Your game crashed!")
		screen = pg.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glClearColor(0.31, 0.39, 0.71, 1.0)

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)

		glBegin(GL_QUADS)
		glVertex2f(-0.7, -0.7)
		glVertex2f(-0.7, 0.7)
		glVertex2f(0.7, 0.7)
		glVertex2f(0.7, -0.7)
		glEnd()

		if Unicode:
			try:
				ASCIILoc = LoadTextures("unicodeL.png")
				TableHeight = 24
			except (pg.error, FileNotFoundError):
				try:
					ASCIILoc = LoadTextures("ascii.png")
					TableHeight = 16
				except (pg.error, FileNotFoundError):
					try:
						ASCIILoc = LoadTextures("../default/unicodeL.png")
						TableHeight = 24
					except (pg.error, FileNotFoundError):
						ASCIILoc = LoadTextures("../default/ascii.png")
						TableHeight = 16
		else:
			try:
				ASCIILoc = LoadTextures("ascii.png")
				TableHeight = 16
			except (pg.error, FileNotFoundError):
				ASCIILoc = LoadTextures("../default/ascii.png")
				TableHeight = 16

		glBindTexture(GL_TEXTURE_2D, ASCIILoc[0])
		UI.write(message, (-0.6, 0.6), 0.05, chardim=(0.65, 1.0), color=(0, 0, 0))
		pg.display.flip()
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		while 1:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					quit()
	except Exception:
		if platform == "win32":
			os.system("start criticalcrash.bat 1 \"" +
					message.replace('"', "'").replace("\n", '" "') + '"')
		quit()