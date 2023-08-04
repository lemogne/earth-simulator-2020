from init import *

def highlight_block(position):
	glBindTexture(GL_TEXTURE_2D, 0)
	glColor3f(0.25, 0.25, 0.25)
	glBegin(GL_LINES)
	for i in Cube.edges:
		for j in Cube.vertices[i]:
			glVertex3fv((np.array((tuple(j * 1.0078125)) + position)))
	glEnd()
	glColor3f(1, 1, 1)

def render_chunk(chunkData):
	s = types[settings.gpu_data_type][3]
	glBindBuffer(GL_ARRAY_BUFFER, chunkData[0])
	glTexCoordPointer(2, types[settings.gpu_data_type][1], 8 * s, (c_void_p)(3 * s))
	glVertexPointer(3, types[settings.gpu_data_type][1], 8 * s, None)
	glNormalPointer(types[settings.gpu_data_type][1], 8 * s, (c_void_p)(5 * s))
	glDrawArrays(GL_TRIANGLES, 0, chunkData[1])

def render_sky(time_start):
	glDisable(GL_DEPTH_TEST)
	World.game_time = settings.starting_time + ((time.time() - time_start) / settings.day_length) * 1024
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glBindTexture(GL_TEXTURE_2D, Textures.sky[0])
	glVertexPointer(3, GL_DOUBLE, 0, Sky.vert_list)
	glTexCoordPointer(2, GL_DOUBLE, 0, Sky.get_tex())
	glNormalPointer(GL_DOUBLE, 0, Sky.normals)
	glDrawArrays(GL_TRIANGLES, 0, int(len(Sky.vert_list)))
	glEnable(GL_DEPTH_TEST)

def render(chat_string):
	#render blocks
	if World.get_block(player.pos + (0, player.height, 0)) == 8:
		glUseProgram(water_shader)
	else:
		glUseProgram(day_night_shader)
	bright_loc = glGetUniformLocation(day_night_shader, "brightness")
	mapsize_loc = glGetUniformLocation(day_night_shader, "mapsize")
	glUniform1f(bright_loc, 1 - Sky.texture_offset(World.game_time))
	glUniform2f(mapsize_loc, *(Textures.mapsize))

	World.load_chunks()

	#Load solid blocks
	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	glMatrixMode(GL_MODELVIEW)
	glPushMatrix()
	glTranslatef(128, 128, 128)
	for reg in World.active_regions:
		for loaded_chunk in reg.loaded_chunks:
			glPushMatrix()
			glTranslatef((loaded_chunk[0] + reg.pos[0]) * World.chunk_size, 0, (loaded_chunk[1] + reg.pos[1]) * World.chunk_size)
			render_chunk(reg.loaded_chunks[loaded_chunk][0])
			glPopMatrix()

	#Load transparent blocks
	glEnable(GL_BLEND)
	for reg in World.active_regions:
		for loaded_chunk in reg.loaded_chunks:
			if reg.loaded_chunks[loaded_chunk][1] != None:
				glPushMatrix()
				glTranslatef((loaded_chunk[0] + reg.pos[0]) * World.chunk_size, 0, (loaded_chunk[1] + reg.pos[1]) * World.chunk_size)
				render_chunk(reg.loaded_chunks[loaded_chunk][1])
				glPopMatrix()
	
	glPopMatrix()
	#Highlight block being looked at
	if (looked_at := get_looked_at()[0]) is not None:
		highlight_block(looked_at // 1)

	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	mode_2D()

	#Draw block overlay if head inside (non-air) block
	if World.get_block(player.pos + (0, player.height, 0)) != 0:
		glBegin(GL_TRIANGLES)
		for i in range(6):
			glTexCoord2fv(Textures.game_blocks[World.get_block(player.pos + (0, player.height, 0))][i])
			glVertex2f(Cube.vertices[Cube.triangles[0][i]][0] * 2 - 1,
			           (Cube.vertices[Cube.triangles[0][i]][1] - 0.5) * (Display.centre[0] / Display.centre[1]) * 2)
		glEnd()

	#Draw currently selected block texture
	glUseProgram(sky_shader)
	if settings.shown:
		if settings.current_block:
			glBegin(GL_TRIANGLES)
			for i in range(6):
				glTexCoord2fv(Textures.game_blocks[settings.current_block][i] / Textures.mapsize)
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

	glDisable(GL_BLEND)
	mode_3D()
