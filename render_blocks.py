from init import *

def render_sky(time_start):
	World.game_time = settings.starting_time + round(((time.time() - time_start) / settings.day_length) * 1024)
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glVertexPointer(3, GL_DOUBLE, 0, Sky.vert_list)
	glTexCoordPointer(2, GL_DOUBLE, 0, Sky.get_tex())
	glDrawArrays(GL_TRIANGLES, 0, int(len(Sky.vert_list)))

def render(chat_string):
	#render blocks
	if World.get_block(player.pos + Vector(0, player.height, 0)) == 8:
		glUseProgram(waterShader)
	else:
		glUseProgram(DayNightShader)
	bright_loc = glGetUniformLocation(DayNightShader, "brightness")
	chunkpos_loc = glGetUniformLocation(DayNightShader, "chunkpos")
	glUniform1f(bright_loc, (clamp(math.cos((World.game_time / 1024) * 2 * math.pi) * 2, -1, 1) + 1) / 2)

	World.load_chunks()

	#Load solid blocks
	for loaded_chunk in World.loaded_chunks:
		glUniform2i(chunkpos_loc, *loaded_chunk)
		World.render_chunk(World.loaded_chunks[loaded_chunk][0])

	#Load transparent blocks
	glEnable(GL_BLEND)
	for loaded_chunk in World.loaded_chunks:
		if World.loaded_chunks[loaded_chunk][1] != None:
			glUniform2i(chunkpos_loc, *loaded_chunk)
			World.render_chunk(World.loaded_chunks[loaded_chunk][1])
	
	glUniform2i(chunkpos_loc, 0, 0)
	#Highlight block being looked at
	if (looked_at := get_looked_at()[0]):
		highlight_block(looked_at // 1)

	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	mode_2D()

	#Draw block overlay if head inside (non-air) block
	if World.get_block(player.pos + Vector(0, player.height, 0)) != 0:
		glBegin(GL_TRIANGLES)
		for i in range(6):
			glTexCoord2fv(Textures.game_blocks[World.get_block(player.pos + Vector(0, player.height, 0))][i])
			glVertex2f(Cube.vertices[Cube.triangles[0][i]][0] * 2 - 1,
			           (Cube.vertices[Cube.triangles[0][i]][1] - 0.5) * (Display.centre[0] / Display.centre[1]) * 2)
		glEnd()

	#Draw currently selected block texture
	glUseProgram(skyShader)
	if settings.shown:
		if settings.current_block:
			glBegin(GL_TRIANGLES)
			for i in range(6):
				glTexCoord2fv(Textures.game_blocks[settings.current_block][i])
				glVertex2f(
					Cube.vertices[Cube.triangles[0][i]][0] * (Display.centre[1] / Display.centre[0]) * settings.icon_size + settings.icon_offset[0],
					Cube.vertices[Cube.triangles[0][i]][1] * settings.icon_size + settings.icon_offset[1])
			glEnd()

		#Write chat
		glBindTexture(GL_TEXTURE_2D, Textures.text[0])
		UI.write(chat_string, (-0.9475, 0.875), 0.03125)

		#Draw crosshair
		glBindTexture(GL_TEXTURE_2D, Textures.cursor[0])
		glBegin(GL_QUADS)
		for i in range(4):
			glTexCoord2fv(character_coords[i])
			glVertex2f(character_coords[i][0] * (Display.centre[1] / Display.centre[0]) * 0.1 - 0.05, character_coords[i][1] * 0.1 - 0.05)
		glEnd()

	#Draw pause menu
	glColor3f(1.0, 1.0, 1.0)
	if UI.paused:
		UI.render_buttons()
		glColor3fv(settings.pause_menu_color)

	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	glDisable(GL_BLEND)
	mode_3D()