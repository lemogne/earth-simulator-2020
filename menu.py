from init import *


#MAIN MENU
def run():
	glDisableClientState(GL_NORMAL_ARRAY)
	UI.buttons = menu_buttons
	glEnable(GL_BLEND)
	mode_2D()
	now = time.time()
	lastFrame = now
	while UI.in_menu:
		now = time.time()
		if not settings.frame_cap or settings.maxFPS * (now - lastFrame) >= 1:
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			glBegin(GL_QUADS)
			for i in range(4):
				glTexCoord2fv(Textures.title_coords[i])
				glVertex2fv(character_coords[i] * 2 - 1)
			glEnd()

			glBindTexture(GL_TEXTURE_2D, Textures.logo[0])
			glBegin(GL_QUADS)
			for i in range(4):
				glTexCoord2fv(character_coords[i])
				glVertex2fv(Textures.logo_coords[i])
			glEnd()

			UI.render_buttons()
			UI.write("v0.2.1", (0.875, -0.98), 0.05, color=(1, 1, 1))
			glBindTexture(GL_TEXTURE_2D, Textures.title[0])
			pg.display.flip()

		UI.check_hover(pg.mouse.get_pos())
		if UI.buttons.is_typing():
			if UI.buttons.get_input_button() == None:
				ChatString = UI.input_text(ChatString, start=2)
				if not UI.buttons.is_typing():
					try:
						exec(ChatString[2:])
					except Exception:
						ChatString = "Invalid Python statement!"
						print("Invalid Python statement!")
			else:
				UI.buttons.get_input_button().run()
				if not UI.buttons.is_typing():
					UI.buttons.set_typing(None)

		for event in pg.event.get():
			if event.type == pg.QUIT:
				quit_game()
			elif event.type == pg.MOUSEBUTTONDOWN:
				if not UI.buttons.get_selected() == None:
					if event.button == 1:
						UI.buttons.get_selected().run()
			elif event.type == pg.VIDEORESIZE and settings.resizeable:
				mode_3D()
				Display.init((event.w, event.h))
				Textures.update_pixel_size()
				mode_2D()
			elif event.type == pg.KEYDOWN and not UI.buttons.is_typing():
				if event.key == pg.K_F2:
					screenshot()

	glBegin(GL_QUADS)
	for i in range(4):
		glTexCoord2fv(Textures.title_coords[i])
		glVertex2fv(character_coords[i] * 2 - 1)
	glEnd()

	glBindTexture(GL_TEXTURE_2D, Textures.text[0])

	UI.write("Loading...", (-0.1625, 0), 0.1, color=(1, 1, 1))

	pg.display.flip()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	glDisable(GL_BLEND)
	mode_3D()
	glEnableClientState(GL_NORMAL_ARRAY)
	pg.mouse.set_visible(False)
	pg.mouse.set_pos(Display.centre)
