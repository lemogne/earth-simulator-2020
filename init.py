import pygame as pg
from pygame.locals import *
import math, time, struct, os, sys, json, re, traceback
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
from ctypes import *
import opensimplex as noise
import settings, menu
from textures import * 
from world import *


# Data Types
BYTE = 0
SHORT = 1
INT = 2
FLOAT = 3
DOUBLE = 4

types = [
	(np.uint8,   GL_BYTE,   c_ubyte,  1,        127),
	(np.uint16,  GL_SHORT,  c_ushort, 2,      32767),
	(np.uint32,  GL_INT,    c_uint32, 4, 2147483647),
	(np.float32, GL_FLOAT,  c_float,  4,          1),
	(np.float64, GL_DOUBLE, c_double, 8,          1)
]

#if settings.gpu_data_type == None:
#	size = max(settings.chunk_size, settings.world_height)
#	if size <= 256:
#		settings.gpu_data_type = BYTE
#	elif size <= 32895:
#		settings.gpu_data_type = SHORT
#	else:
#		settings.gpu_data_type = INT
settings.gpu_data_type = FLOAT

make_coord_array()

chat_string = ""


class Sky:
	triangles = (
		(2, 0, 5), (0, 3, 5), (1, 2, 5), (3, 1, 5),
		(6, 8, 4), (9, 6, 4), (8, 7, 4), (7, 9, 4),
		(6, 2, 8), (6, 0, 2), (9, 0, 6), (9, 3, 0),
		(8, 1, 7), (8, 2, 1), (7, 3, 9), (7, 1, 3)
	)
	vertices = (
		(-1, 1, -1), (1, 1, 1), (-1, 1, 1), (1, 1, -1), (0, -1, 0), 
		(0, 4 / 3, 0), (-1, 0, -1), (1, 0, 1), (-1, 0, 1), (1, 0, -1)
	)
	vert_list = np.array(vertices)[np.array(triangles).ravel()]
	normals = np.zeros(len(vert_list))


	def init():
		t = 0.05
		b = 0.95
		m = (2 * t + b) / 3
		Sky.tex_list = np.array(
			[(0, m), (0, m), (0, t)] * 4 + 
			[(0, b), (0, b), (0, b)] * 4 +
			[(0, b), (0, m), (0, b), (0, b), (0, m), (0, m)] * 4
		)


	def get_tex():
		tempS = Sky.texture_offset(World.game_time)
		return Sky.tex_list + (tempS, 0)


	def texture_offset(time):
		hardness = 2
		return (clamp(-math.cos((time / 1024) * 2 * math.pi) * hardness, -0.9, 0.9) + 1) / 2


class Time:
	time_start = time.time()
	last_tick = time_start
	last_frame = time_start

	# FPS counting
	last_second = int(time_start)
	frames = 0
	prev_frames = 0


def process_chunks(skip_smoothing = False):
	for reg in World.active_regions:
		#print(reg.to_be_loaded)
		while (ch := reg.to_be_loaded.pop(0) if reg.to_be_loaded else None): # atomic assignment to `ch`
			while not skip_smoothing and settings.min_FPS and time.time() - Time.last_frame >= 1 / settings.min_FPS:
				#print("idle render")
				if menu.UI.in_menu:
					return
				time.sleep(0.05)
			
			reg.preloaded_data[ch] = World.process_chunk(ch + reg.pos)
			#print("render", ch + reg.pos)
			World.new_chunks += 1


def chunk_thread():
	try:
		while not menu.UI.in_menu:
			process_chunks()
			time.sleep(0.05)
	except Exception as e:
		World.thread_exception = e
		print("".join(traceback.format_exc()))


def get_schematic(file):
	raw_json = open(f"schematics/{settings.schematic_pack}/{file}.json").read()
	schem = json.loads(raw_json)
	schem_blocks = np.array(schem[0])
	return (schem_blocks, compute_lighting(schem_blocks), np.array(schem[1]))


def get_looked_at():
	def rnd(p, dx):
		return (dx < 0) * np.floor(p) - (dx > 0) * np.floor(-p) - p

	r_pos = player.pos + (0, player.height, 0)
	o_pos = r_pos
	dt = 0
	nm = np.array(-player.norm)
	invnm = 1 / nm
	
	while np.linalg.norm(r_pos - (player.pos + (0, player.height, 0))) <= settings.hand_reach:
		block = World.get_block(r_pos)
		
		if block not in non_highlightable:
			hitbox = models[block_models[block]].hitbox
			if in_hitbox(r_pos, hitbox):
				return (r_pos, o_pos)
			
			minim = rnd((r_pos - hitbox[0]) / (hitbox[1] - hitbox[0]), nm / (hitbox[1] - hitbox[0])) * invnm
			minim = (minim * (hitbox[1] - hitbox[0])) + hitbox[0]
		else:
			minim = rnd(r_pos, nm) * invnm
		dt = float(min(minim[minim != 0]))
		o_pos = r_pos + (0, 0, 0)
		r_pos -= dt * 1.1 * player.norm

	return (None, None)


def set_time(time):
	World.starting_time += time - World.game_time


def clamp(x, bottom, top):
	return max(bottom, min(top, x))


def screenshot():
	buffer = glReadPixels(0, 0, *Display.size, GL_RGBA, GL_UNSIGNED_BYTE)
	screen_surf = pg.image.fromstring(buffer, Display.size, "RGBA", True)
	
	try:
		pg.image.save(screen_surf, f"{settings.screenshot_dir}/{int(time.time())}.png")
	except (pg.error, FileNotFoundError):
		os.mkdir(settings.screenshot_dir)
		pg.image.save(screen_surf, f"{settings.screenshot_dir}/{int(time.time())}.png")


def init_pygame():
	pg.init()
	pg.display.init()
	pg.display.set_icon(pg.image.load("icon.ico"))
	if sys.platform == "win32":
		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("earthsimV0.2")
	pg.display.set_caption("Earth Simulator 2020")


def init_schematics():
	global schematic
	raw_json = open(f"schematics/{settings.schematic_pack}/schematics.json").read()
	schematic_names = json.loads(raw_json)
	schematic = dict()
	for schem in schematic_names:
		schematic[schem] = [get_schematic(x) for x in schematic_names[schem]]


init_pygame()
day_night_shader, sky_shader, water_shader = Display.init(settings.nominal_res)
game_blocks, seethrough, biometint = Textures.init()
World.init(game_blocks, seethrough, biometint)
Textures.update_pixel_size()
init_schematics()
Sky.init()

