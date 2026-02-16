import math, time, struct, os, sys, json, re, traceback
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
import numpy as np
import pygame as pg
from pygame.locals import *
import settings


character_coords = np.array(((0, 1), (0, 0), (1, 0), (1, 1)))


def load_shaders():
	vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/shader.vert", 'r').read(), GL_VERTEX_SHADER)
	fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/shader.frag", 'r').read(), GL_FRAGMENT_SHADER)
	sky_vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/skysh.vert", 'r').read(), GL_VERTEX_SHADER)
	sky_fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/skysh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	water_vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/watersh.vert", 'r').read(), GL_VERTEX_SHADER)
	#water_fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/watersh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	
	day_night_shader = glCreateProgram()
	glAttachShader(day_night_shader, vertex_shader)
	glAttachShader(day_night_shader, fragment_shader)
	glLinkProgram(day_night_shader)

	sky_shader = glCreateProgram()
	glAttachShader(sky_shader, sky_vertex_shader)
	glAttachShader(sky_shader, sky_fragment_shader)
	glLinkProgram(sky_shader)

	water_shader = glCreateProgram()
	glAttachShader(water_shader, water_vertex_shader)
	glAttachShader(water_shader, fragment_shader)
	glLinkProgram(water_shader)
	
	return day_night_shader, sky_shader, water_shader


class Display:
	def init(size):
		# Initialising screen, 3D rendering and position/rotation vectors
		if settings.fullscreen:
			if sys.platform == "win32":
				Display.size = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
			else:
				import subprocess
				output = subprocess.Popen(
					'xrandr | grep "*" | cut -d" " -f4', shell=True,
					stdout=subprocess.PIPE
				).communicate()[0]
				output = output.decode()[:-1].split("x")
				Display.size = (int(output[0]), int(output[1]))
		else:
			Display.size = size
		
		Display.fovX = math.degrees(
		    2 * math.atan(math.tan(
				math.radians(settings.fov_Y / 2)) * (Display.size[0] / Display.size[1]
			))
		)
		Display.centre = (Display.size[0] // 2, Display.size[1] // 2)

		Display.screen = pg.display.set_mode(
			Display.size, 
			(FULLSCREEN * settings.fullscreen) | DOUBLEBUF | OPENGL | (pg.RESIZABLE * settings.resizeable)
		)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluPerspective(
			settings.fov_Y, 
			(Display.size[0] / Display.size[1]), 0.1,
		    settings.render_distance * 3**0.5 * settings.chunk_size
		)
		
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		glEnableClientState(GL_NORMAL_ARRAY)
		glClearColor(0.0, 0.0, 0.0, 0.0)
		return load_shaders()



class Textures:
	def init():
		game_blocks = json.loads(open(f"textures/{settings.texture_pack}/texturing.json").read())
		seethrough = np.array(json.loads(open(f"textures/{settings.texture_pack}/transparency.json").read()))
		biometint = np.array(json.loads(open(f"textures/{settings.texture_pack}/biometint.json").read()))
		Textures.ui = Textures.load("UI.png")
		
		if unicode:
			try:
				Textures.text = Textures.load("unicodeL.png")
				Textures.texttable_height = 24
			except (pg.error, FileNotFoundError):
				Textures.text = Textures.load("ascii.png")
				Textures.texttable_height = 16
		else:
			Textures.text = Textures.load("ascii.png")
			Textures.texttable_height = 16
		
		Textures.terrain = Textures.load("textures.png")
		Textures.sky = Textures.load("sky.png", GL_LINEAR)
		Textures.logo = Textures.load("logo.png")
		Textures.title = Textures.load("title.png")
		Textures.cursor = Textures.load("cursor.png")
		Textures.biomes = Textures.load("biomes.png", GL_LINEAR, GL_CLAMP_TO_EDGE)

		title_size = (Textures.title[1].get_width(), Textures.title[1].get_height())
		Textures.text_ratio = Textures.text[1].get_width() * (
			Textures.texttable_height / 16
		) / Textures.text[1].get_height()
		TSR = title_size[1] / title_size[0]
		Textures.title_coords = character_coords * np.array([TSR * (Display.centre[0] / Display.centre[1]), 1])
		logo_size = (Textures.logo[1].get_width(), Textures.logo[1].get_height())
		Textures.logo_coords = (character_coords * 2 - 1) * np.array(
		    (TSR, logo_size[1] / logo_size[0])) * settings.logo_scale + np.array([0.0625 * settings.logo_scale, 0.65])

		Textures.mapsize_big = (Textures.terrain[1].get_width(), Textures.terrain[1].get_height())
		Textures.mapsize = np.array(Textures.mapsize_big) / 16
		Textures.game_blocks = Textures.generate(game_blocks)
		
		return game_blocks, seethrough, biometint

	def update_pixel_size():
		Textures.pixel_size = 2 * Textures.text[1].get_height() / (Display.size[1] * Textures.texttable_height)


	def generate(blocks):
		Face = np.array([
			[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 2, 2, 2], 
			[0, 1, 1, 1, 2, 3, 2], [0, 1, 2, 2, 3, 4, 3], [0, 1, 2, 3, 4, 5, 4]
		])
		tex_array = []
		for block in blocks:
			cube_sides = np.tile(Cube.sides, (7, 1))
			tiles = np.array(block)[Face[len(block) - 1]]
			block_textures = np.repeat(
			    np.array([tiles % Textures.mapsize[0], -(tiles + 1) // Textures.mapsize[0]]).T, 6, 0)
			tex_array.append(cube_sides + block_textures)
		return np.array(tex_array)


	def load(file, filter = GL_NEAREST, clamping = GL_REPEAT):
		texture_surface = pg.image.load(f"textures/{settings.texture_pack}/{file}")
		texture_data = pg.image.tostring(texture_surface, "RGBA", 1)

		glEnable(GL_TEXTURE_2D)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		texid = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, texid)
		glTexImage2D(
			GL_TEXTURE_2D, 0, GL_RGBA, texture_surface.get_width(), 
			texture_surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data
		)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, clamping)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clamping)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter)
		return (texid, texture_surface)


class Cube:
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0)))
	edges = np.array(((0, 1), (0, 3), (0, 4), (1, 2), (1, 7), (2, 5), (2, 3), (3, 6), (4, 6), (4, 7), (5, 6), (5, 7)))
	quads = np.array(((5, 7, 1, 2), (3, 0, 4, 6), (2, 1, 0, 3), (6, 4, 7, 5), (7, 4, 0, 1), (5, 2, 3, 6)))
	triangles = (np.array((5, 7, 1, 5, 1, 2)), np.array((3, 0, 4, 3, 4, 6)), np.array((2, 1, 0, 2, 0, 3)), np.array((6, 4, 7, 6, 7, 5)),
	             np.array((7, 4, 0, 7, 0, 1)), np.array((5, 2, 3, 5, 3, 6)), None)
	sides = np.array(((1, 0), (1, 1), (0, 1), (1, 0), (0, 1), (0, 0)))
	normals = (np.array((0, 0, 1)), np.array((0, 0, -1)), np.array((-1, 0, 0)), np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, -1, 0)), None)
	is_side_full = np.array((True, True, True, True, True, True))
	hitbox = np.array(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))


class Slab(Cube):
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 0.5, 1), (1, 0.5, 0), (1, 0, 0), (1, 0, 1), (0, 0.5, 1), (0, 0, 0), (0, 0, 1), (0, 0.5, 0)))
	triangles = (np.array((5, 7, 1, 5, 1, 2)), np.array((3, 0, 4, 3, 4, 6)), np.array((2, 1, 0, 2, 0, 3)), np.array((6, 4, 7, 6, 7, 5)),
	             None, np.array((5, 2, 3, 5, 3, 6)), np.array((7, 4, 0, 7, 0, 1)))
	normals = (np.array((0, 0, 1)), np.array((0, 0, -1)), np.array((-1, 0, 0)), np.array((1, 0, 0)), None, np.array((0, -1, 0)), np.array((0, 1, 0)))
	is_side_full = np.array((False, False, False, False, True, False))
	hitbox = np.array(((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)))


class Liquid(Cube):
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 0.875, 1), (1, 0.875, 0), (1, 0, 0), (1, 0, 1), (0, 0.875, 1), (0, 0, 0), (0, 0, 1), (0, 0.875, 0)))
	is_side_full = np.array((False, False, False, False, True, False))
	hitbox = np.array(((0.0, 0.0, 0.0), (1.0, 0.875, 1.0)))


class Ice(Liquid):
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 0.9375, 1), (1, 0.9375, 0), (1, 0, 0), (1, 0, 1), (0, 0.9375, 1), (0, 0, 0), (0, 0, 1), (0, 0.9375, 0)))
	is_side_full = np.array((False, False, False, False, True, False))
	hitbox = np.array(((0.0, 0.0, 0.0), (1.0, 0.9375, 1.0)))


class Layer(Slab):
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 0.125, 1), (1, 0.125, 0), (1, 0, 0), (1, 0, 1), (0, 0.125, 1), (0, 0, 0), (0, 0, 1), (0, 0.125, 0)))
	hitbox = np.array(((0.0, 0.0, 0.0), (1.0, 0.125, 1.0)))


class Air(Cube):
	# Faces: front, back, right, left, top, bottom
	#vertices = np.array(((1, 0.875, 1), (1, 0.875, 0), (1, -0.125, 0), (1, -0.125, 1), (0, 0.875, 1), (0, -0.125, 0), (0, -0.125, 1), (0, 0.875, 0)))
	triangles = (None, None, None, None, None, None, None)


class Shrub(Cube):
	vertices = np.array(((1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0)))
	triangles = (np.array((6, 4, 1, 6, 1, 2)), np.array((2, 1, 4, 2, 4, 6)), np.array((5, 7, 0, 5, 0, 3)), np.array((3, 0, 7, 3, 7, 5)), None, None, None)
	normals = (np.array((1, 0, -1)) / (0.5 ** 0.5), np.array((-1, 0, 1)) / (0.5 ** 0.5), np.array((-1, 0, 1)) * (0.5 ** 0.5), np.array((1, 0, -1)) / (0.5 ** 0.5), None, None, None)
	is_side_full = np.array((False, False, False, False, False, False))

