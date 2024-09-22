import pygame as pg
from pygame.locals import *
import math, time, struct, os, sys, json, re, traceback
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
from ctypes import *
import opensimplex as noise
import settings

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

if settings.gpu_data_type == None:
	size = max(settings.chunk_size, settings.world_height)
	if size <= 256:
		settings.gpu_data_type = BYTE
	elif size <= 32895:
		settings.gpu_data_type = SHORT
	else:
		settings.gpu_data_type = INT

# Menu Options


def quit_game(event):
	glDisableClientState(GL_VERTEX_ARRAY)
	glDisable(GL_TEXTURE_2D)
	glDisableClientState(GL_TEXTURE_COORD_ARRAY)
	pg.quit()
	quit()


def toggle_menu(event):
	global paused_buttons
	UI.buttons = paused_buttons
	UI.paused ^= True
	pg.mouse.set_visible(UI.paused)


class Button_Box:
	type = np.array([
	    [
	        ((0, 1), (1, 1), (1, 0.875), (0, 0.875)),  #Long Button
	        ((0, 0.75), (1, 0.75), (1, 0.625), (0, 0.625))
	    ],
	    [
	        ((0, 0.625), (1, 0.625), (1, 0.5), (0, 0.5)),  #Long Textbox
	        ((0, 0.375), (1, 0.375), (1, 0.25), (0, 0.25))
	    ],
	    [
	        ((0, 0.875), (0.5, 0.875), (0.5, 0.75), (0, 0.75)),  #Short Button
	        ((0.5, 0.875), (1, 0.875), (1, 0.75), (0.5, 0.75))
	    ],
	    [
	        ((0, 0.5), (0.5, 0.5), (0.5, 0.375), (0, 0.375)),  #Short Textbox
	        ((0.5, 0.5), (1, 0.5), (1, 0.375), (0.5, 0.375))
	    ]
	])
	vertices_long = np.array(((-1, 0.125), (1, 0.125), (1, -0.125), (-1, -0.125)))
	vertices_short = np.array(((-0.5, 0.125), (0.5, 0.125), (0.5, -0.125), (-0.5, -0.125)))


class Interface:

	def __init__(self, buttons, main_menu=False):
		self.buttons = buttons
		self.input_button = None
		self.menu_mode = main_menu
		self.typing = False
		self.selected = None

	def screen(self):
		UI.buttons = self.buttons

	def back(self):
		global paused_buttons, menu_buttons
		UI.buttons = menu_buttons if self.menu_mode else paused_buttons

	def get_button(self, button):
		return self.buttons[button]

	def set_input_button(self, button):
		self.input_button = button

	def get_input_button(self):
		if self.input_button != None:
			return self.buttons[self.input_button]
		return None

	def select(self, button):
		self.selected = button

	def get_selected(self):
		if self.selected != None:
			return self.buttons[self.selected]
		return None

	def is_selected(self, button):
		if type(button) is str:
			return self.selected == button
		elif self.selected != None:
			return self.buttons[self.selected] == button
		else:
			return False

	def is_typing(self):
		return self.typing

	def set_typing(self, val):
		self.typing = val

	def __getitem__(self, a):
		return self.buttons[a]

	def __iter__(self):
		return iter(self.buttons.keys())

	def __add__(self, a):
		return Interface({**self.buttons, **a.buttons})


class UI:
	buttons = Interface({})
	char_sizes = [None, None]
	in_menu = True
	paused = False
	show_game_info = False

	def init_font():
		if os.path.exists(f"textures/{settings.texture_pack}/font.json"):
			raw_json = open(f"textures/{settings.texture_pack}/font.json").read()
			char_size_dict = json.loads(raw_json)
		else:
			char_size_dict = {"\n" : [0, -1]}
		default_size = char_size_dict["default"] if "default" in char_size_dict else [1, 4]
		UI.char_sizes[0] = np.full(Textures.texttable_height * 16, default_size[0] / 8, dtype=np.float64)
		UI.char_sizes[1] = np.full(Textures.texttable_height * 16, (default_size[1] + 2) / 8, dtype=np.float64)

		for char in char_size_dict:
			if len(char) != 1:
				continue
			if ord(char) > Textures.texttable_height * 16:
				print(f"Warning: character {char} defined in texturepack {settings.texture_pack} outside valid range")
				continue
			UI.char_sizes[0][ord(char)] = char_size_dict[char][0] / 8
			UI.char_sizes[1][ord(char)] = (char_size_dict[char][1] + 2) / 8


	def write(text, loc, size, color=(1, 1, 1), shadow=False):
		if text == "":
			return

		if settings.text_align:
			loc = justify_text(loc)
			size = round(size / Textures.pixel_size) * Textures.pixel_size

		def rcount(a):
			# Source: StackOverflow
			without_reset = np.insert((UI.char_sizes[1][a][:-1] - UI.char_sizes[0][a][1:]).cumsum(), 0, 0)
			reset_at = (a == ord('\n'))
			overcount = np.maximum.accumulate(without_reset * reset_at)
			result = without_reset - overcount - 1
			return result

		loc = np.array(loc)
		text_array = np.char.expandtabs(text, 4).reshape(1).view(np.int32)
		text_array = text_array[(text_array < Textures.texttable_height * 16)]
		printable = text_array >= 32

		column = rcount(text_array)
		line = -(text_array == ord('\n')).cumsum()
		text_array = text_array[printable]
		column = column[printable]
		line = line[printable]

		column_line_array = np.array((column, line)).T
		column_line_tiled = np.tile(column_line_array, (1, 4)).reshape((len(text_array), 4, 2))

		char_index = np.array((text_array & 15, ((Textures.texttable_height << 4) - 1 - text_array) >> 4)).T
		tiled_chars = np.tile(char_index, (1, 4)).reshape((len(text_array), 4, 2))
		tex = ((tiled_chars + character_coords) / (16, Textures.texttable_height)).ravel()
		vert = (loc + ((character_coords + column_line_tiled) * size) *
		        ((Display.centre[1] / Display.centre[0]) * Textures.text_ratio, 1)).ravel()

		if shadow:
			shadowloc = loc - (1.0, 1.0) / np.array(Display.centre)
			shadowvert = (shadowloc + ((character_coords + column_line_tiled) * size) *
					((Display.centre[1] / Display.centre[0]) * Textures.text_ratio, 1)).ravel()
			glColor3fv(np.array(color) * 0.3)
			glBindBuffer(GL_ARRAY_BUFFER, 0)
			glVertexPointer(2, GL_DOUBLE, 0, shadowvert)
			glTexCoordPointer(2, GL_DOUBLE, 0, tex)
			glDrawArrays(GL_QUADS, 0, int(len(tex) / 2))

		glColor3fv(color)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glVertexPointer(2, GL_DOUBLE, 0, vert)
		glTexCoordPointer(2, GL_DOUBLE, 0, tex)
		glDrawArrays(GL_QUADS, 0, int(len(tex) / 2))
		glColor3fv((1, 1, 1))

	def input_text(variable, event, start=0, stop=None):
		UI.buttons.set_typing(True)
		if event.type == pg.KEYDOWN:
			key = event.key
			if key == 8:
				if len(variable) > start:
					variable = variable[:-1]
				return variable
			elif key == 13:
				UI.buttons.set_typing(False)
				UI.buttons.set_input_button(None)
				return variable
			elif key == 27:
				UI.buttons.set_typing(False)
				UI.buttons.set_input_button(None)
				return variable[:start]
			if stop == None or len(variable) <= stop:
				variable = variable + event.unicode
		return variable

	def check_hover(m_pos):
		m_posGL = (( m_pos[0] - Display.centre[0]) / Display.centre[0],
		           (-m_pos[1] + Display.centre[1]) / Display.centre[1])
		for button_ in UI.buttons:
			button = UI.buttons[button_]
			box = button.get_box()
			b1 = ((box[0] * settings.button_scale)[0] * (Display.centre[1] / Display.centre[0]),
			      (box[0] * settings.button_scale)[1])
			b2 = ((box[2] * settings.button_scale)[0] * (Display.centre[1] / Display.centre[0]),
			      (box[2] * settings.button_scale)[1])
			if b1[0] < m_posGL[0] < b2[0] and b1[1] > m_posGL[1] > b2[1]:
				UI.buttons.select(button_)
				break
		else:
			UI.buttons.select(None)

	def render_buttons():
		glBindTexture(GL_TEXTURE_2D, Textures.ui[0])
		glBegin(GL_QUADS)
		for button in UI.buttons:
			if UI.buttons[button].has_texture():
				texture = UI.buttons[button].get_texture()
				box = UI.buttons[button].get_box()
				for i in range(4):
					glTexCoord2fv(texture[int(UI.buttons.is_selected(button))][i])
					glVertex2f((box[i] * settings.button_scale)[0] * (Display.centre[1] / Display.centre[0]),
					           (box[i] * settings.button_scale)[1])
		glEnd()

		glBindTexture(GL_TEXTURE_2D, Textures.text[0])
		for button_ in UI.buttons:
			button = UI.buttons[button_]
			box = button.get_box()
			if button.has_texture() and button.is_text_box():
				UI.write(button.get_text() + ("_" if UI.buttons.get_input_button() == button else ""),
				         ((box[0] * settings.button_scale + 0.1)[0] * (Display.centre[1] / Display.centre[0]),
				          (box[0] * settings.button_scale - 0.1)[1]),
				         0.05,
				         color=(0, 0, 0))
			else:
				UI.write(button.get_text(),
				         ((box[0] * settings.button_scale + 0.1)[0] * (Display.centre[1] / Display.centre[0]),
				          (box[0] * settings.button_scale - 0.1)[1]), 0.05, color=(1, 1, 1), shadow=True)

	def type_button(name, event):
		UI.buttons.set_input_button(name)
		UI.buttons[name].set_text(UI.input_text(UI.buttons[name].get_text(), event))


class Button:

	def __init__(self, position, text, input_mode=None, short=False, function=None):
		if input_mode == None:
			self.texture = None
		else:
			if short == None:
				short = False
			self.texture = Button_Box.type[2 * short + input_mode]
		if short:
			self.box = Button_Box.vertices_short
		else:
			self.box = Button_Box.vertices_long
		self.input_mode = input_mode
		# In-place operator overwrites the ButtonBox class because pointers
		self.box = self.box + np.array(position)
		self.text = text
		self.function = function
		self.UI = None

	def run(self, event):
		if self.function != None:
			self.function(event)

	def add_to_UI(self, UI):
		self.UI = UI

	def get_box(self):
		return self.box

	def input(self):
		UI.setInputButton(self)

	def has_texture(self):
		return self.texture is not None

	def get_texture(self):
		return self.texture

	def get_text(self):
		return self.text

	def set_text(self, text):
		self.text = text

	def is_text_box(self):
		return self.input_mode

	def type_in(self, event):
		self.text = UI.input_text(self.text, event)


def leave_world(event):
	# Unloads chunks and quits the world
	save_world()
	World.regions = {}
	World.active_regions = {}
	World.loaded_chunks = {}
	World.preloaded_chunks = {}
	UI.paused = False
	UI.in_menu = True
	World.init()


class Start_Game:

	def run(event):
		World.new = True
		# Determining world seed:
		# if field is empty, generate random seed; else, try to interpret seed as number. If not possible, display error.
		if UI.buttons["Seed"].get_text() == "":
			World.seed = int((time.time() % 1) * 10000000)
		else:
			try:
				World.seed = int(UI.buttons["Seed"].get_text())
			except ValueError:
				UI.buttons["Info"].set_text("World seed must be an integer!")
				return

		worldname = UI.buttons["Name"].get_text()
		if worldname == "":
			worldname = "world_" + time.strftime("%Y-%m-%d_%H%M%S")
		filepath = f"worlds/{worldname}.esw"
		if os.path.exists(filepath):
			UI.buttons["Info"].set_text(f"World '{worldname}' already exists!")
			return
		World.file = open(filepath, "w+b")  # Create/overwrite world file as given by user input

		UI.in_menu = False

	def screen(event):
		UI.buttons = Start_Game.buttons

	def get_seed(event):
		UI.buttons.set_input_button("Seed")
		UI.buttons["Seed"].set_text(UI.input_text(UI.buttons["Seed"].get_text(), event))

	def get_name(event):
		UI.buttons.set_input_button("Name")
		UI.buttons["Name"].set_text(UI.input_text(UI.buttons["Name"].get_text(), event))

	def back(event):
		global menu_buttons
		UI.buttons = menu_buttons

	buttons = Interface({
	    "WorldName": Button((-1.5, 0.5), "World Name:", None, True),
	    "Name": Button((0, 0.5), "", True, False, get_name),
	    "WorldSeed": Button((-1.5, 0.2), "World Seed:", None, True),
	    "Seed": Button((0, 0.2), "", True, False, get_seed),
	    "CreateWorld": Button((0, -0.2), "Create World", False, False, run),
	    "Info": Button((0, -0.4), ""),
	    "Back": Button((0, -0.7), "Back", False, False, back)
	})


bytesneeded = lambda x: np.int8(math.log(x, 256) + 1 // 1)

def save_world():
	global game_blocks
	try:
		raw_region_data = {}
		for reg in World.region_table:
			if reg not in World.regions:
				raw_region_data[reg] = Load_World.load_region_raw(reg)

		savefile = World.file
		savefile.seek(0, 0)
		savefile.truncate(0)
		savefile.write(b"ES20\x00v0.4")  # Write file header containing game version

		write_bytes = lambda data, bytes: savefile.write(np.array(data).astype(f'V{bytes}').tobytes())

		# Calculating bytes needed to save world data; saving that information to the file
		BLblocks = bytesneeded(len(game_blocks))

		savefile.write(np.array([BLblocks, World.infinite == 0, World.region_size], dtype=np.int8).tobytes())
		savefile.write(np.array([World.height, World.chunk_size], dtype=np.int32).tobytes())
		savefile.write(np.array(player.pos, dtype=np.float32).tobytes())
		savefile.write(np.array(player.rot, dtype=np.float32).tobytes())
		savefile.write(np.array([World.game_time, World.seed, World.water_level], dtype=np.int32).tobytes())
		savefile.write(np.array(World.heightlim, dtype=np.int32).tobytes())

		savefile.write(np.array(World.T_res, dtype=np.float32).tobytes())
		savefile.write(np.array(World.HL_res, dtype=np.float32).tobytes())
		savefile.write(np.array(World.G_res, dtype=np.float32).tobytes())
		savefile.write(np.array(World.C_res, dtype=np.float32).tobytes())
		savefile.write(np.array(World.B_res, dtype=np.float32).tobytes())

		savefile.write(np.array(World.tree_res, dtype=np.float32).tobytes())
		savefile.write(np.array([World.tree_density_mean, World.tree_density_var, 0, 0, 0], dtype=np.float32).tobytes())
		savefile.write(np.array(len(World.regions) + len(raw_region_data), dtype=np.int32).tobytes())

		table_pos = savefile.tell()

		savefile.seek(16 * (len(World.regions) + len(raw_region_data)), 1)
		i = savefile.tell()

		# Save unloaded chunks first
		for region in raw_region_data:
			savefile.seek(table_pos)
			savefile.write(np.array(region, dtype=np.int32).tobytes())
			savefile.write(np.array([i, raw_region_data[region][0]], dtype=np.int32).tobytes())
			table_pos = savefile.tell()
			savefile.seek(i)
			savefile.write(raw_region_data[region][1])
			i = savefile.tell()

		# Save each chunk separately, in sequence
		for region in World.regions:
			savefile.seek(table_pos)
			savefile.write(np.array(region, dtype=np.int32).tobytes())
			region = World.regions[region]
			savefile.write(np.array([i, len(region.chunks)], dtype=np.int32).tobytes())
			table_pos = savefile.tell()
			savefile.seek(i)

			for ch in region.chunks:
				pg.event.get()  # To prevent game window from freezing
				savefile.write(np.array(ch, dtype=np.uint8).tobytes())  # Writes the chunk coordinates to the file

				# Numpy RLE from stackoverflow
				flat_chunk = region.chunks[ch].transpose((1, 0, 2)).flatten()
				loc_run_start = np.empty(len(flat_chunk), dtype=bool)
				loc_run_start[0] = True
				np.not_equal(flat_chunk[:-1], flat_chunk[1:], out=loc_run_start[1:])
				run_starts = np.nonzero(loc_run_start)[0]
				run_values = flat_chunk[loc_run_start]
				run_lengths = np.diff(np.append(run_starts, len(flat_chunk)))

				# Block counting loop; compresses block data by making a list of (amount of blocks in a row, block ID); saves to file
				for counter, block in zip(run_lengths, run_values):
					BLpos = bytesneeded(counter)
					if counter > 127:
						write_bytes(BLpos + 127, 1)
					write_bytes(counter, BLpos)
					write_bytes(block, BLblocks)
			i = savefile.tell()
				

		UI.buttons["Info"].set_text("World saved successfully!")
	except Exception as e:
		UI.buttons["Info"].set_text(f"Failed to save world: {e}")
	print(UI.buttons["Info"].get_text())


class Load_World:

	def run(event):
		global player, game_blocks, make_coord_array
		# Clear Chunk and Light arrays (dictionaries, whatever)
		player.old_chunkpos = None
		World.regions = {}
		World.active_regions = {}
		World.regions_to_load = []
		UI.buttons["Info"].set_text("Loading...")
		World.new = False

		# Check if a world has been selected
		if Load_World.selected_world:
			name = Load_World.selected_world
		else:
			UI.buttons["Info"].set_text("No world selected!")
			print("No world selected!")
			return

		# Check if world actually exists and open it
		try:
			readfile = open(f"worlds/{name}.esw", "r+b")
		except FileNotFoundError:
			UI.buttons["Info"].set_text("No such world!")
			print("No such world!")
			return

		try:
			# check if the file is actually a world, and if it is the right version
			magic_const = readfile.read(4)
			readfile.seek(1, 1)
			version = readfile.read(4)
			if magic_const != b"ES20":
				UI.buttons["Info"].set_text("This file is not a world!")
				print("This file is not a world!")
				return
			if version == b"v0.4":
				World.bytes_for_block_ID = struct.unpack("b", readfile.read(1))[0]
				World.infinite = struct.unpack("b", readfile.read(1))[0] == 0
				World.region_size = struct.unpack("b", readfile.read(1))[0]
				World.height = struct.unpack("i", readfile.read(4))[0]
				World.chunk_size = struct.unpack("I", readfile.read(4))[0]

				make_coord_array()
				player.pos = np.array((struct.unpack("3f", readfile.read(12))))
				player.rot = np.array((struct.unpack("3f", readfile.read(12))))
				World.game_time = struct.unpack("i", readfile.read(4))[0]

				# World gen
				World.seed = struct.unpack("i", readfile.read(4))[0]
				World.water_level = struct.unpack("i", readfile.read(4))[0]
				World.heightlim = struct.unpack("2i", readfile.read(8))
				noise.seed(int(World.seed))

				World.T_res = struct.unpack("2f", readfile.read(8))
				World.HL_res = struct.unpack("2f", readfile.read(8))
				World.G_res = struct.unpack("2f", readfile.read(8))
				World.C_res = struct.unpack("2f", readfile.read(8))
				World.B_res = struct.unpack("2f", readfile.read(8))

				World.tree_res = struct.unpack("2f", readfile.read(8))
				World.tree_density_mean = struct.unpack("f", readfile.read(4))[0]
				World.tree_density_var = struct.unpack("f", readfile.read(4))[0]
				readfile.seek(12, 1)

				# Region table
				region_table_size = struct.unpack("I", readfile.read(4))[0]
				World.region_table = dict()
				for _ in range(region_table_size):
					region_coords = struct.unpack("2i", readfile.read(8))
					file_data = struct.unpack("2i", readfile.read(8))
					World.region_table[region_coords] = file_data
				
				# Load region where player is
				player.chunkpos = player.pos // World.chunk_size
				region = tuple((player.chunkpos // World.region_size)[[0, 2]])
				readfile.seek(0, 0)
				World.file = readfile
				World.regions[region] = Region(region)
				Load_World.load_region(region)
				World.regions_to_load = []
				player.old_rot = None
				player.old_pos = None

			elif version == b"v0.3":
				UI.buttons["Info"].set_text(
					f"This world is from version {str(version, 'ASCII')} and uses an old format. "\
					"Please convert it by re-saving it in this version. "\
					"Worlds in this format will not be supported much longer. "
				)
				print(UI.buttons["Info"].get_text())
				World.bytes_for_block_ID = struct.unpack("b", readfile.read(1))[0]
				World.infinite = struct.unpack("b", readfile.read(1))[0] == 0
				World.region_size = struct.unpack("b", readfile.read(1))[0]
				World.height = struct.unpack("i", readfile.read(4))[0]
				World.chunk_size = struct.unpack("I", readfile.read(4))[0]

				make_coord_array()
				player.pos = np.array((struct.unpack("3f", readfile.read(12))))
				player.rot = np.array((struct.unpack("3f", readfile.read(12))))
				World.game_time = struct.unpack("i", readfile.read(4))[0]

				# World gen
				World.seed = struct.unpack("i", readfile.read(4))[0]
				World.water_level = struct.unpack("i", readfile.read(4))[0]
				World.heightlim = struct.unpack("2i", readfile.read(8))
				noise.seed(int(World.seed))
 
				World.T_res = struct.unpack("2f", readfile.read(8))
				World.HL_res = struct.unpack("2f", readfile.read(8))
				readfile.seek(8, 1)
				World.G_res = struct.unpack("2f", readfile.read(8))

				World.tree_res = struct.unpack("2f", readfile.read(8))
				World.tree_density_mean = struct.unpack("f", readfile.read(4))[0]
				World.tree_density_var = struct.unpack("f", readfile.read(4))[0]
				readfile.seek(12, 1)

				# Region table
				region_table_size = struct.unpack("I", readfile.read(4))[0]
				World.region_table = dict()
				for _ in range(region_table_size):
					region_coords = struct.unpack("2i", readfile.read(8))
					file_data = struct.unpack("2i", readfile.read(8))
					World.region_table[region_coords] = file_data
				
				# Load region where player is
				player.chunkpos = player.pos // World.chunk_size
				region = tuple((player.chunkpos // World.region_size)[[0, 2]])
				readfile.seek(0, 0)
				World.file = readfile
				World.regions[region] = Region(region)
				Load_World.load_region(region)
				World.regions_to_load = []
				player.old_rot = None
				player.old_pos = None

			elif version in [b"v0.1", b"v0.2"]:
				UI.buttons["Info"].set_text(
					f"This world is from version {str(version, 'ASCII')} and uses an old format. "\
					"Please convert it by re-saving it in this version. "\
					"Worlds in this format will not be supported much longer. "
				)
				print(UI.buttons["Info"].get_text())
				readfile.seek(0, 0)
				data = readfile.read()
				readfile.close()

				# The world file contains the amount of bytes needed to write certain data (i.e. 1 byte to save a block ID); This data is read here
				BLblocks = data[10]
				BLpos = data[11]
				BLch = data[12]
				BLheight = data[13]
				World.chunk_size = struct.unpack("i", data[15:19])[0]
				World.infinite = False

				# World height, position and camera rotation are read here
				World.height = struct.unpack("i", data[19:23])[0]
				make_coord_array()
				player.pos = np.array((struct.unpack("3f", data[27:39])))
				player.rot = np.array((struct.unpack("3f", data[39:51])))

				# Chunk reading loop (reads until end of block data flag is read)
				i = 51
				World.region_table = dict()
				while data[i:i + 6] != b"\x00\x00\x00\x00\x00\x00":
					pg.event.get()  #prevents window from freezing
					ChBuffer = []

					chunk = (int.from_bytes(data[i:i + BLch], "little", signed=True), 
					         int.from_bytes(data[i + BLch:i + 2 * BLch], "little", signed=True))  # Chunk position
					i += BLch * 2

					# reads blocks until chunk end flag is read
					while data[i:i + 4] != b"\x00\x00\x00\x00":
						block = int.from_bytes(data[i + BLpos:i + BLpos + BLblocks], "little")
						ChBuffer += [block] * int.from_bytes(data[i:i + BLpos], "little")
						i += BLpos + BLblocks

					# Tries shaping blocks read into a chunk shape; if that is impossible, the the chunk must be malformed and hence the file corrupted
					try:
						region, ch = World.get_region(chunk)
						if not region:
							reg_coords = (chunk[0] // settings.region_size, chunk[1] // settings.region_size)
							region = Region(reg_coords)
							World.regions[reg_coords] = region
						region.chunks[ch] = np.reshape(np.array(ChBuffer),
													(World.chunk_size, World.height, World.chunk_size)).astype(np.uint8)
					except ValueError:
						UI.buttons["Info"].set_text("World file corrupted!")
						print("World file corrupted!")
						return

					for y in range(World.height):
						if seethrough[region.chunks[ch][:, y, :]].any():
							region.chunk_min_max[ch] = (y / World.chunk_size, 0)
							break
					for y in range(World.height - 1, -1, -1):
						if (region.chunks[ch][:, y, :] != 0).any():
							chmin = region.chunk_min_max[ch][0]
							region.chunk_min_max[ch] = (chmin, (y / World.chunk_size) - chmin)
							break

					i += 4

					# Reads lighting data
					region.light[ch] = np.reshape(
						np.frombuffer(np.frombuffer(data[i:i + (World.chunk_size**2) * BLheight],
													dtype=(f"V{BLheight}")).astype("V4"),
									dtype=np.int32), (World.chunk_size, World.chunk_size))
					i += (World.chunk_size**2) * BLheight

					# Set chunk as generated
					region.gen_chunks[ch] = True

					# Check if chunk end flag is present; if not, the file must be corrupted
					if data[i:i + 4] != b"\x00\x00\x00\x00":
						UI.buttons["Info"].set_text("World file corrupted!")
						print("World file corrupted!")
						return
					i += 4
				i += 6

				# Read important world information
				World.seed = struct.unpack("f", data[i:i + 4])[0]
				i += 4
				settings.tree_density_mean = struct.unpack("f", data[i:i + 4])[0]
				i += 4
				settings.starting_time = struct.unpack("i", data[i:i + 4])[0]
			else:
				UI.buttons["Info"].set_text(f"The version of the world, {str(version, 'ASCII')} is not compatible with this version!")
				print(UI.buttons["Info"].get_text())
				return
			UI.in_menu = False
			UI.paused = False

			UI.buttons["Info"].set_text("World loaded successfully!")
		except Exception as e:
			UI.buttons["Info"].set_text(f"Failed to load world: {e}")
			print("".join(traceback.format_exc()))
		print(UI.buttons["Info"].get_text())

	def load_region(reg):
		if reg not in World.region_table:
			World.new = True	# To force terrain generation if just loaded world
			return
		region = World.regions[reg]
		readfile = World.file
		readfile.seek(World.region_table[reg][0])
		BLblocks = World.bytes_for_block_ID
		chunk_length = World.chunk_size**2 * World.height

		# Chunk reading loop (reads until end of block data flag is read)
		for _ in range(World.region_table[reg][1]):
			pg.event.get()  # prevents window from freezing

			chunk_buffer = np.zeros(chunk_length)

			ch = (int.from_bytes(readfile.read(1), "little", signed=True), 
			      int.from_bytes(readfile.read(1), "little", signed=True))  # Chunk position

			# reads blocks until chunk is filled
			i = 0
			while i < chunk_length:
				k = int.from_bytes(readfile.read(1), "little")
				if k > 127:
					n = int.from_bytes(readfile.read(k - 127), "little")
				else:
					n = k
				block = int.from_bytes(readfile.read(BLblocks), "little")
				chunk_buffer[i:i+n] = block
				i += n

			region.chunks[ch] = np.reshape(
				np.array(chunk_buffer), 
				(World.height, World.chunk_size, World.chunk_size)
			).astype(np.uint8).transpose((1, 0, 2))

			coords = np.array(reg) * World.region_size + ch

			x = np.arange(World.chunk_size) + coords[1] * World.chunk_size
			z = np.arange(World.chunk_size) + coords[0] * World.chunk_size

			World.gen_biomemap(tuple(coords), x, z)

			# Compute chmin, chmax
			for y in range(World.height):
				if seethrough[region.chunks[ch][:, y, :]].any():
					chmin_norm = y
					region.chunk_min_max[ch] = (y / World.chunk_size, 0)
					break
			for y in range(World.height - 1, -1, -1):
				if (region.chunks[ch][:, y, :] != 0).any():
					chmin = region.chunk_min_max[ch][0]
					chmax_norm = y
					region.chunk_min_max[ch] = (chmin, (y / World.chunk_size) - chmin)
					break

			# Compute lighting data
			region.light[ch] = compute_lighting(region.chunks[ch][:, chmin_norm:chmax_norm + 2, :]) + chmin_norm

			# Set chunk as generated
			region.gen_chunks[ch] = True


	def load_region_raw(reg):
		readfile = World.file
		readfile.seek(World.region_table[reg][0])
		BLblocks = World.bytes_for_block_ID
		
		region_buffer = b""
		chunk_length = World.chunk_size**2 * World.height

		# Chunk reading loop (reads until end of block data flag is read)
		for _ in range(World.region_table[reg][1]):
			#pg.event.get()  # prevents window from freezing

			region_buffer += readfile.read(2)
			# reads blocks until chunk is filled
			i = 0
			while i < chunk_length:
				kr = readfile.read(1)
				region_buffer += kr
				k = int.from_bytes(kr, "little")
				if k > 127:
					nr = readfile.read(k - 127)
					region_buffer += nr
					n = int.from_bytes(nr, "little")
				else:
					n = k
				blockr = readfile.read(BLblocks)
				region_buffer += blockr
				i += n
		return (World.region_table[reg][1], region_buffer)


	def screen(event):
		try:
			worldlist = os.listdir("worlds")
			worlds = list(filter(lambda x: (x[-4:] == ".esw"), worldlist))
			worlds.sort(key=lambda x: -os.path.getmtime("worlds/" + x))
			Load_World.worlds = [x[:-4] for x in worlds]

		except FileNotFoundError:
			os.mkdir("worlds")
			Load_World.worlds = []
			Load_World.default_buttons["Info"].set_text("Worlds save directory not found!")

		Load_World.pages = []
		for i in range((len(Load_World.worlds) + 3) // settings.worlds_per_page):
			page = {}
			page_length = settings.worlds_per_page
			if i == len(Load_World.worlds) // settings.worlds_per_page:
				if len(Load_World.worlds) % settings.worlds_per_page != 0:
					page_length = len(Load_World.worlds) % settings.worlds_per_page
			for j in range(page_length):
				page[j] = Button((0, 0.4 - j * 0.3), Load_World.worlds[settings.worlds_per_page * i + j], False, False,
				                 Load_World.gen_func(settings.worlds_per_page * i + j))
			Load_World.pages.append(Interface(page))
		if len(Load_World.pages) == 0:
			Load_World.pages.append(Interface({}))
		Load_World.reload()
		UI.buttons = Load_World.buttons

	def back(event):
		global menu_buttons
		UI.buttons = menu_buttons

	def gen_func(world):

		def f(event):
			for i in range(len(Load_World.pages)):
				page_length = settings.worlds_per_page
				if i == len(Load_World.worlds) // settings.worlds_per_page:
					if len(Load_World.worlds) % settings.worlds_per_page != 0:
						page_length = len(Load_World.worlds) % settings.worlds_per_page
				for j in range(page_length):
					if settings.worlds_per_page * i + j != world:
						Load_World.pages[i][j].texture = Button_Box.type[0]
						Load_World.pages[i][j].input_mode = False
					else:
						Load_World.pages[i][j].texture = Button_Box.type[1]
						Load_World.pages[i][j].input_mode = True
				if Load_World.page == i:
					Load_World.reload()
			Load_World.selected_world = Load_World.worlds[world]

		return f

	def next_page(event):
		if Load_World.page < len(Load_World.pages) - 1:
			Load_World.page += 1
			Load_World.reload()

	def prev_page(event):
		if Load_World.page > 0:
			Load_World.page -= 1
			Load_World.reload()

	def reload():
		Load_World.buttons = Load_World.default_buttons + Load_World.pages[Load_World.page]
		Load_World.buttons["Page"].set_text(f"Page {Load_World.page + 1}")
		UI.buttons = Load_World.buttons

	default_buttons = Interface({
	    "Title": Button((-0.5, 0.7), "Worlds", None, True),
	    "Page": Button((0.5, 0.7), "Page 1", None, True),
	    "Info": Button((0, 0.1 - settings.worlds_per_page * 0.3), "", None),
	    "Load": Button((0.5, 0.4 - settings.worlds_per_page * 0.3), "Load World", False, True, run),
	    "Back": Button((-0.5, 0.4 - settings.worlds_per_page * 0.3), "Back", False, True, back),
	    "Prev": Button((-1.5, 0.4 - settings.worlds_per_page * 0.3), "Previous", False, True, prev_page),
	    "Next": Button((1.5, 0.4 - settings.worlds_per_page * 0.3), "Next", False, True, next_page)
	})
	buttons = default_buttons
	pages = []
	page = 0
	selected_world = None
	worlds = []


class Settings:
	variables = {}
	graphics_changed = False

	def main(event):
		settings.logo_shown = True
		UI.buttons = Settings.buttons

	def cancel(event):
		global paused_buttons, menu_buttons
		settings.logo_shown = True
		Settings.variables = {}
		if UI.in_menu:
			UI.buttons = menu_buttons
		else:
			UI.buttons = paused_buttons

	def apply(event):
		setfile = open("settings.py", "r")
		setlines = setfile.readlines()
		setfile.close()
		setfile = open("settings.py", "w")
		for line in setlines:
			line_parse = re.fullmatch(r"(?:([a-zA-Z_0-9]+) *= *(.+?))?([ \t]*#.*)?\n?", line)
			
			if line_parse and (var := line_parse.group(1)) in Settings.variables:
				comment = line_parse.group(3)
				comment = comment if comment else ''
				setfile.write(f"{var} = {Settings.variables[var]}{comment}\n")
				setattr(settings, var, eval(Settings.variables[var]))
			else:
				setfile.write(line)
		imm = UI.in_menu
		if len(Settings.variables) > 0:
			global unicode, game_blocks, seethrough
			if Settings.graphics_changed:
				glDeleteTextures(
					5,
					np.array([Textures.ui, Textures.logo, Textures.text, Textures.terrain, Textures.title])[:, 0])
				if not imm:
					for region in World.regions:
						World.regions[region].unload_vram()
					player.rot = np.array((0.0, 0.0, 0.0))
				pg.quit()
				init_pygame()
				Display.init(settings.nominal_res)
				Textures.init()
				Textures.update_pixel_size()
				UI.init_font()
				Sky.init()
			init_schematics()
			player.init()

			if imm:
				World.init()
				if Settings.graphics_changed:
					mode_2D()
			elif Settings.graphics_changed:
				World.load_chunks(True)
				process_chunks(True)

		Sky.init()
		setfile.close()
		Settings.cancel(event)

	def update_variable(name, var, type_):
		if not UI.buttons.is_typing():
			if type_ == str:
				Settings.variables[var] = '"' + UI.buttons[name].get_text() + '"'
			elif type_ == tuple:
				if var not in Settings.variables:
					value = getattr(settings, var)
				else:
					value = eval(Settings.variables[var])
				Settings.variables[var] = "("
				for i in range(len(value)):
					if i != int(name[-1]):
						Settings.variables[var] += str(value[i]) + ", "
					else: 
						Settings.variables[var] += UI.buttons[name].get_text() + ", "
				Settings.variables[var] = Settings.variables[var][:-2] + ")"
			else:
				Settings.variables[var] = UI.buttons[name].get_text()

	def generate_ui():
		def datatype(data):
			"""Heuristic function to determine datatype from string"""
			if '"' in data or "'" in data:
				return str
			elif data in ["True", "False"]:
				return bool
			elif "(" in data:
				return tuple
			elif "." in data or "/" in data:
				return float
			return int

		def gen_name(var_name : str):
			"""Generates userfriendly name from variable name"""
			return var_name.capitalize().replace("_", " ")

		# Function generator functions
		# Used to generate functions for buttons with given parameters
		def gen_cat_func(category):
			def cat_func(event):
				UI.buttons = Settings.categories[category]
				settings.logo_shown = False
			return cat_func

		def gen_bool_func(var_name, button_name, is_graphics):
			def button_func(event):
				attr = getattr(settings, var_name)
				Settings.variables[var_name] = str(attr ^ True)
				UI.buttons[button_name].set_text(str(attr ^ True))
				setattr(settings, var_name, attr ^ True)
				if is_graphics:
					Settings.graphics_changed = True
			return button_func

		def gen_right_tuple_func(button_name, var_name, var_type, is_graphics):
			def button_func(event):
				UI.type_button(button_name + "1", event)
				Settings.update_variable(button_name + "1", var_name, var_type)
				if is_graphics:
					Settings.graphics_changed = True
			return button_func
		
		def gen_left_tuple_func(button_name, var_name, var_type, is_graphics):
			def button_func(event):
				UI.type_button(button_name + "0", event)
				Settings.update_variable(button_name + "0", var_name, var_type)
				if is_graphics:
					Settings.graphics_changed = True
			return button_func

		def get_button_func(button_name, var_name, var_type, is_graphics):
			def button_func(event):
				UI.type_button(button_name, event)
				Settings.update_variable(button_name, var_name, var_type)
				if is_graphics:
					Settings.graphics_changed = True
			return button_func 

		# Open and read settings file
		setfile = open("settings.py", "r")
		setlines = setfile.readlines()
		setfile.close()

		current_category = ""

		# Button coordinates
		y = -0.9
		x = -0.5
		cat_y = 0.6

		# Main generation loop
		for line in setlines:
			# Parse line into (variable) = (value)   #(comment)
			line_parse = re.fullmatch(r"(?:([a-zA-Z_0-9]+) *= *(.+?))?[ \t]*(?:#(.*))?\n?", line)
			if not line_parse:
				continue
			comment = line_parse.group(3)
			var_name = line_parse.group(1)
			value = line_parse.group(2)

			# Create / generate settings categories from comments
			# skip lines with no variables being set
			if not var_name:
				if comment:
					current_category = comment.capitalize()
					Settings.buttons.buttons[current_category] = Button((0, cat_y), current_category, False, False, gen_cat_func(current_category))
					Settings.categories[current_category] = Interface({"Back": Button((0.67, -1.2), "Back", False, True, Settings.main)})
					cat_y -= 0.3
					y = -0.9
					x = -1
				continue

			# Skip hidden settings
			if comment and comment[:6] == "hidden":
				continue

			button_name = var_name + "_value"
			var_type = datatype(value)

			# Assign correct behaviour of input field given datatype
			if var_type is bool:
				button_func = gen_bool_func(var_name, button_name, current_category == "Graphics")
			elif var_type is tuple:
				# Match tuple
				tuple_values = re.search(r"\(([\-0-9.]*) *, *([\-0-9.]*)\)", value)

				if not tuple_values:
					# This happens when the tuple has more than two values or something is malformed
					# This is complicated to handle and will be left out for now
					continue

				# Special case: right value of 2-tuple
				right_val = tuple_values.group(2)
				right_button_func = gen_right_tuple_func(button_name, var_name, var_type, current_category == "Graphics")
				Settings.categories[current_category].buttons[button_name + "1"] = Button((x, y), right_val, True, True, right_button_func)

				# left value adjusted so that rest of function can carry on as normal
				button_func = gen_left_tuple_func(button_name, var_name, var_type, current_category == "Graphics")
				button_name += "0"
				y += 0.25
				value = tuple_values.group(1)
			else:
				button_func = get_button_func(button_name, var_name, var_type, current_category == "Graphics")

			# If string, extract value
			if var_type is str:
				value = re.sub("[\"'](.*)[\"']", r"\1", value)	

			# Insert new setting into correct category
			Settings.categories[current_category].buttons[var_name+"_label"] = Button((x - 1, y), gen_name(var_name), None, True)
			Settings.categories[current_category].buttons[button_name] = Button((x, y), value, var_type != bool, True, button_func)

			# Coordinate calculations
			y += 0.3
			if y > 1.2:
				y = -0.9
				x = 1.5

	buttons = Interface({
	    "Cancel": Button((0.67, -1.2), "Cancel", False, True, cancel),
	    "OK": Button((-0.67, -1.2), "OK", False, True, apply)
	})

	categories = {}


paused_buttons = Interface({
	"Info": Button((0, -1.2), ""),
    "Main": Button((0, -0.8), "Back to Main Menu", False, False, leave_world),
    "Save": Button((0, -0.4), "Save World", False, False, save_world),
    "Settings": Button((0, 0), "Settings", False, False, Settings.main),
    "Resume": Button((0, 0.4), "Resume Game", False, False, toggle_menu)
})

menu_buttons = Interface({
	"Info": Button((0, -1.2), ""),
    "Quit": Button((0, -0.9), "Quit Game", False, False, quit_game),
    "Settings": Button((0, -0.5), "Settings", False, False, Settings.main),
    "Load": Button((0, -0.1), "Load World", False, False, Load_World.screen),
    "New": Button((0, 0.3), "New World", False, False, Start_Game.screen)
})



class Cube:
	# Faces: front, back, right, left, top, bottom
	vertices = np.array(((1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0)))
	edges = np.array(((0, 1), (0, 3), (0, 4), (1, 2), (1, 7), (2, 5), (2, 3), (3, 6), (4, 6), (4, 7), (5, 6), (5, 7)))
	quads = np.array(((5, 7, 1, 2), (3, 0, 4, 6), (2, 1, 0, 3), (6, 4, 7, 5), (7, 4, 0, 1), (5, 2, 3, 6)))
	triangles = np.array(((5, 7, 1, 5, 1, 2), (3, 0, 4, 3, 4, 6), (2, 1, 0, 2, 0, 3), (6, 4, 7, 6, 7, 5),
	                      (7, 4, 0, 7, 0, 1), (5, 2, 3, 5, 3, 6)))
	sides = np.array(((1, 0), (1, 1), (0, 1), (1, 0), (0, 1), (0, 0)))
	normals = np.array(((0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)))


def collision_check(pos, ds, dt):
	def collide(i, pos, ds):
		if i == 1:
			offset = 0 if ds[i] > 0 else (1 - (settings.player_height % 1))
		else:
			offset = settings.player_width
		offset += settings.hitbox_epsilon
		if ds[i] < 0:
			pos[i] = math.ceil(pos[i]) - offset
		elif ds[i] > 0:
			pos[i] = math.floor(pos[i]) + offset
		ds[i] = 0

	# Check for block collisions
	segments = math.ceil(np.linalg.norm(ds * dt))
	for _ in range(segments):
		# Basic check in each dimension separately
		for i in range(3):
			if player.check_in_block(i, dt / segments, ds, pos):
				collide(i, pos, ds)

		# Edge cases
		while player.check_in_block(-1, dt / segments, ds, pos) and ds.any():
			collide(abs(ds).argmax(), pos, ds)
		pos -= ds * dt / segments


class Player:
	pos = np.array((0.0, 0.0, 0.0))
	mv = np.array((0.0, 0.0, 0.0))
	rot = np.array((0.0, 0.0, 0.0))  # pitch, yaw, roll
	norm = np.array((0.0, 0.0, 0.0))

	# Efficiency redundancies
	old_chunkpos = None
	old_rot = None

	# Used in the rendering engine to prevent sudden 'snapping' when mv = 0
	old_pos = pos

	def init(self):
		self.chunkpos = self.pos // settings.chunk_size
		self.height = settings.player_height
		self.flying = settings.flying

	def do_tick(self, dt):
		if not (UI.paused or UI.buttons.is_typing()):
			self.old_pos = self.pos + (0, 0, 0)
			block_under = World.get_block(self.pos - (0, 0.01, 0))
			# Calculate movement vector based on key presses and environment
			keystates = pg.key.get_pressed()
			accel = np.array((0.0, 0.0, 0.0))
			forward = (keystates[pg.K_w] - keystates[pg.K_s])
			sideways = (keystates[pg.K_a] - keystates[pg.K_d])
			downward = (keystates[pg.K_LSHIFT] - keystates[pg.K_SPACE])
			if forward and sideways:
				forward *= 0.707106781188
				sideways *= 0.707106781188
			accel[0] = forward * self.norm[0] + sideways * self.norm[2]
			accel[1] = downward * settings.jump_height * (bool(block_under != 0) or self.flying or self.mv[1] == 0)
			accel[2] = forward * self.norm[2] - sideways * self.norm[0]
			if self.flying:
				accel *= (settings.flying_speed / settings.movement_speed)
				self.mv[1] = 0
			else:
				accel[1] += dt * settings.gravity

			friction = 0.6
			if block_under == 14:
				friction = 0.05
			elif block_under == 0:
				friction = 0.2

			density = 1
			in_water = (World.get_block(self.pos) == 8)
			if in_water:
				density = 0.5

			self.mv[0] = self.mv[0] * (1 - friction) + accel[0] * friction
			self.mv[1] = (self.mv[1] + accel[1]) * density
			self.mv[2] = self.mv[2] * (1 - friction) + accel[2] * friction

			collision_check(self.pos, self.mv, dt)

			# MOVEMENT
			self.chunkpos = self.pos // World.chunk_size

	def not_in_hitbox(self, block):
		"""Checks if block at given location is not in hitbox of player."""
		hitbox_min = self.pos - (settings.player_width, 0, settings.player_width)
		hitbox_max = self.pos + (settings.player_width, settings.player_height, settings.player_width)
		block_min = np.floor(block)
		block_max = block_min + 1

		return ((hitbox_max < block_min) | (block_max < hitbox_min)).any()


	def check_in_block(self, dim, dt, mv, pos):
		# TODO: optimise!!!
		hitbox_min = pos - (settings.player_width, 0, settings.player_width)
		hitbox_max = pos + (settings.player_width, settings.player_height, settings.player_width)

		# dim < 0 means check in all dimensions
		# dim ≥ 0 means only check that dimension
		if dim >= 0:
			if mv[dim] > 0:
				hitbox_min[dim] -= mv[dim] * dt
				hitbox_max[dim] = hitbox_min[dim]
			else:
				hitbox_max[dim] -= mv[dim] * dt
				hitbox_min[dim] = hitbox_max[dim]
		else:
			for i in range(3):
				if mv[i] > 0:
					hitbox_min[i] -= mv[i] * dt
				else:
					hitbox_max[i] -= mv[i] * dt
			
		x_min = math.floor(hitbox_min[0])
		x_max = math.floor(hitbox_max[0])
		y_min = math.floor(hitbox_min[1])
		y_max = math.floor(hitbox_max[1])
		z_min = math.floor(hitbox_min[2])
		z_max = math.floor(hitbox_max[2])

		for x in range(x_min, x_max + 1):
			for y in range(y_min, y_max + 1):
				for z in range(z_min, z_max + 1):
					if (block := World.get_block((x, y, z))) != 0 and block != 8:
						return True
		return False

	def rotate(self, mouse_pos):
		if not (UI.paused or UI.buttons.is_typing()):
			if not mouse_pos == Display.centre:
				m_x = mouse_pos[0] - Display.centre[0]
				m_y = Display.centre[1] - mouse_pos[1]
				rv = np.array((
					max(
						min(m_y * settings.mouse_sensitivity, 90 - self.rot[0]), 
						-90 - self.rot[0]
					),
					m_x * settings.mouse_sensitivity, 0
				))
				self.rot += rv
				pg.mouse.set_pos(Display.centre)
			else:
				rv = np.array((0.0, 0.0, 0.0))

			# norm is the normal vector of the culling plane, also the 'forward' vector
			self.norm = np.array((
				-settings.movement_speed * math.sin(math.radians(self.rot[1])),
				-settings.movement_speed * math.tan(math.radians(self.rot[0])),
				 settings.movement_speed * math.cos(math.radians(self.rot[1]))
			))


player = Player()
player.init()

class Region:
	preloaded_chunks = {}
	loaded_chunks = {}
	to_be_loaded = []
	preloaded_data = {}
	chunk_min_max = {}
	chunks = {}
	light = {}
	pos = np.array((0, 0))
	chunk_coords = None
	in_view = None
	chunk_y_lims = None
	gen_chunks = None

	def __init__(self, pos):
		self.preloaded_chunks = dict()
		self.loaded_chunks = dict()
		self.to_be_loaded = list()
		self.preloaded_data = dict()
		self.chunk_min_max = np.full((World.region_size, World.region_size, 2), 0.0)
		self.chunks = dict()
		self.light = dict()
		self.pos = np.array(pos) * World.region_size
		self.chunk_coords = None
		self.gen_chunks = np.full((World.region_size, World.region_size), False)

		if pos in World.region_table:
			World.regions_to_load.append(pos)


	def __del__(self):
		for ch in self.preloaded_chunks:
			glDeleteBuffers(1, int(self.preloaded_chunks[ch][0][0]))
			if self.preloaded_chunks[ch][1] != None:
				glDeleteBuffers(1, int(self.preloaded_chunks[ch][1][0]))

	def unload_vram(self):
		self.__del__()
		self.preloaded_chunks = dict()
		self.loaded_chunks = dict()
		self.preloaded_data = dict()

	def load_chunks(self, change_pos, change_rot, force_load=False):
		reg_pos = tuple(self.pos // World.region_size)
		if reg_pos in World.regions_to_load:
			if force_load:
				Load_World.load_region(reg_pos)
				World.regions_to_load.remove(reg_pos)
			else:
				return
		if change_pos or self.chunk_coords is None or force_load or World.new_chunks >= 3:
			self.chunk_coords = np.mgrid[0:World.region_size, 0:World.region_size].T[:, :, ::-1]
			chunk_distance = settings.chunk_distance(
				(abs(self.chunk_coords[:, :, 0] - player.chunkpos[0] + self.pos[0]),
				abs(self.chunk_coords[:, :, 1] - player.chunkpos[2] + self.pos[1]))
			)
			self.chunk_coords = self.chunk_coords[chunk_distance <= settings.render_distance]
			gen_status = self.gen_chunks[chunk_distance <= settings.render_distance]
			if World.infinite:
				to_generate = self.chunk_coords[~gen_status]
			self.chunk_coords = self.chunk_coords[gen_status]
			self.chunk_y_lims = self.chunk_min_max[chunk_distance <= settings.render_distance][gen_status]
			player.old_rot = None
			if World.infinite:
				for ch in to_generate:
					key = tuple((ch + self.pos).astype(np.int32))
					if key not in World.chunks_to_generate:
						World.chunks_to_generate.append(key)
		if force_load:
			self.in_view = np.full(shape=len(self.chunk_coords), fill_value=True)
			player.old_rot = None
		elif change_pos or change_rot or World.new_chunks >= 3:
			self.in_view = World.chunk_in_view(self.chunk_coords + self.pos, self.chunk_y_lims)
		else:
			return
		World.new_chunks = 0
		self.loaded_chunks = dict()
		while len(self.preloaded_data) > 0:
			ch, data = self.preloaded_data.popitem()
			self.loaded_chunks[ch] = World.load_chunk(data)
			self.preloaded_chunks[ch] = self.loaded_chunks[ch]

		self.to_be_loaded = list()
		for ch in self.chunk_coords[self.in_view]:
			ch = tuple(ch.astype(np.int32))
			if not ch in self.preloaded_chunks.keys():
				if not ch in self.preloaded_data.keys():
					self.to_be_loaded.append(ch)
			else:
				self.loaded_chunks[ch] = self.preloaded_chunks[ch]
		self.to_be_loaded.sort(key = lambda x: settings.chunk_distance(self.pos + x - player.chunkpos[[0, 2]]))
		

	# TODO: possible error: sudden jump in y level between neighbouring chunks
	# can lead to rendering errors -> fix! (idea: involve surrounding blocks)
	def thorough_chmin(self, ch):
		for y in range(World.height):
			if seethrough[self.chunks[ch][:, y, :]].any():
				return y / World.chunk_size
		return World.height

	def thorough_chmax(self, ch):
		for y in range(World.height - 1, -1, -1):
			if seethrough[self.chunks[ch][:, y, :]].any():
				chmin = self.chunk_min_max[ch][0]
				return (y / World.chunk_size) - chmin
		return 0

class World:
	seed = 0
	new = False
	game_time = 0
	thread_exception = None
	height = settings.world_height
	chunk_size = settings.chunk_size
	heightlim = settings.heightlim
	water_level = settings.water_level
	region_size = settings.region_size
	infinite = settings.infinite
	T_res = settings.T_res
	HL_res = settings.HL_res
	B_res = settings.B_res
	G_res = settings.G_res		
	C_res = settings.C_res

	tree_density_mean = settings.tree_density_mean
	tree_density_var = settings.tree_density_var
	tree_res = settings.tree_res
	heightmap = {}
	biomemap = {}
	blockmap = {}
	trees = {}
	regions = {}
	regions_to_load = []
	chunks_to_generate = []
	active_regions = []
	file = None
	region_table = {}
	bytes_for_block_ID = 0
	new_chunks = 0

	coord_array = []
	coord_array3 = []

	def init():
		World.height = settings.world_height
		World.chunk_size = settings.chunk_size
		World.game_time = settings.starting_time
		World.heightlim = settings.heightlim
		World.water_level = settings.water_level
		World.region_size = settings.region_size
		World.T_res = settings.T_res
		World.HL_res = settings.HL_res
		World.B_res = settings.B_res
		World.G_res = settings.G_res
		World.C_res = settings.C_res

		World.tree_density_mean = settings.tree_density_mean
		World.tree_density_var = settings.tree_density_var
		World.tree_res = settings.tree_res
		World.infinite = settings.infinite

		World.file = None
		World.region_table = {}
		World.regions_to_load = []
		World.bytes_for_block_ID = bytesneeded(len(game_blocks))
		World.new_chunks = 0

	def gen_biomemap(t_coords, x, z):
		#World.biomemap[t_coords] = np.sin(1.57 * noise.noise2array(x / World.B_res[0] + 73982.98, z / World.B_res[1] + 43625.87))
		World.biomemap[t_coords] = noise.noise2array(x / World.B_res[0] + 73982.98, z / World.B_res[1] + 43625.87)


	def load_chunks(ForceLoad = False):
		reg_max = (player.chunkpos + settings.render_distance) // World.region_size
		reg_min = (player.chunkpos - settings.render_distance) // World.region_size
		change_pos = (player.old_chunkpos != player.chunkpos).any()
		change_rot = (player.old_rot != player.rot // 5).any()
		if change_pos:
			player.old_chunkpos = player.chunkpos + (0, 0, 0)
		if change_rot:
			player.old_rot = player.rot // 5
		World.active_regions = []
		for i in range(int(reg_min[0]), int(reg_max[0]) + 1):
			for j in range(int(reg_min[2]), int(reg_max[2]) + 1):
				if (i, j) not in World.regions:
					if not World.infinite:
						continue
					World.regions[(i, j)] = Region((i, j))
				World.regions[(i, j)].load_chunks(change_pos, change_rot, ForceLoad)
				World.active_regions.append(World.regions[(i, j)])
		World.active_regions.sort(key = lambda x: settings.chunk_distance(x.pos - player.chunkpos[[0, 2]]))

	def load_chunk(chunkdata):
		vert_tex_list = chunkdata[0][0]
		counter = chunkdata[0][1]
		vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo)
		glBufferData(
			GL_ARRAY_BUFFER,
			len(vert_tex_list) * types[settings.gpu_data_type][3],
			(types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list),
			GL_STATIC_DRAW
		)

		if chunkdata[1] != None:
			vert_tex_list = chunkdata[1][0]
			counter_transp = chunkdata[1][1]
			vbo_transp = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, vbo_transp)
			glBufferData(
				GL_ARRAY_BUFFER,
				len(vert_tex_list) * types[settings.gpu_data_type][3],
				(types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list), 
				GL_STATIC_DRAW
			)
			return ((vbo, counter), (vbo_transp, counter_transp))
		return ((vbo, counter), None)

	def get_region(chunkpos):
		reg_coords = (chunkpos[0] // World.region_size, chunkpos[1] // World.region_size)
		if reg_coords in World.regions:
			reg = World.regions[reg_coords]
		else:
			reg = None
		ch = (int(chunkpos[0] % World.region_size), int(chunkpos[1] % World.region_size))
		return (reg, ch)

	def get_hum_temp(latitude, y):
		# 3 * π ≈ 9.425
		a = 3.5
		b = 0.8
		c = 0.9
		hum = (np.cos(9.425 * latitude) + 1) * ( -latitude**2 + 1) / 2
		temp = c * (np.exp(-(a * latitude + b)**2) + np.exp(-(a * latitude - b)**2)) - (y - 40) / 200
		return np.clip(np.column_stack((hum, temp)), 0, 1) * types[settings.gpu_data_type][4]

	def process_chunk(chunkpos):
		region, ch = World.get_region(chunkpos)
		chunk = region.chunks[ch]
		blocks = np.vstack(chunk)
		chunk_light = region.light[ch]
		chunk_biome = np.vstack(
			np.reshape(
				np.tile(World.biomemap[tuple(chunkpos)], World.height), 
				(World.chunk_size, World.height, World.chunk_size)
			)
		)
		
		# TODO: add check to not render chunks whose neighbours haven't been generated yet

		# Shifts 3D block array by +/-1 in each direction to determine neighbour
		neighbours = [
		    np.dstack(((World.chunk_data((chunkpos[0], chunkpos[1] - 1))[:, :, -1:]), chunk[:, :, :-1])),
		    np.dstack((chunk[:, :, 1:], (World.chunk_data((chunkpos[0], chunkpos[1] + 1))[:, :, 0:1]))),
		    np.vstack((chunk[1:, :, :], (World.chunk_data((chunkpos[0] + 1, chunkpos[1]))[0:1, :, :]))),
		    np.vstack(((World.chunk_data((chunkpos[0] - 1, chunkpos[1]))[-1:, :, :]), chunk[:-1, :, :])),
		    np.pad(chunk, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=constV)[:, 1:, :],
		    np.pad(chunk, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=constV)[:, :-1, :]
		]
		neighbours_light = [
		    np.hstack(((World.light_data((chunkpos[0], chunkpos[1] - 1))[:, -1:]), chunk_light[:, :-1])),
		    np.hstack((chunk_light[:, 1:], (World.light_data((chunkpos[0], chunkpos[1] + 1))[:, 0:1]))),
		    np.vstack((chunk_light[1:, :], (World.light_data((chunkpos[0] + 1, chunkpos[1]))[0:1, :]))),
		    np.vstack(((World.light_data((chunkpos[0] - 1, chunkpos[1]))[-1:, :]), chunk_light[:-1, :])),
		    chunk_light - 1, chunk_light + 1
		]
		verts = []
		tex_verts = []
		biome_verts = []
		normals = []
		transp_verts = []
		transp_tex_verts = []
		transp_biome_verts = []
		transp_normals = []
		counter = 0
		counter_transp = 0
		for i in range(6):
			# Basically, for each of the 6 possible faces of all cubes, we filter out all those, whose neighbour is not air;
			# the remainder we turn into vertex and texture data
			neighbours[i] = np.vstack(neighbours[i]).astype(int)
			light_neighb = np.vstack(
				np.reshape(
					np.tile(neighbours_light[i], World.height), 
					(World.chunk_size, World.height, World.chunk_size)
				)
			)
			neighb_transp = seethrough[neighbours[i]]

			blocks_transp = blocks[neighb_transp]
			coords_transp = World.coord_array[neighb_transp]
			light_transp = light_neighb[neighb_transp]
			biome_transp = chunk_biome[neighb_transp]

			solid_mask = ~seethrough[blocks_transp]
			blocks_show = blocks_transp[solid_mask]  # Solid blocks
			coords_show = coords_transp[solid_mask]
			light_show = light_transp[solid_mask]
			biome_show = biome_transp[solid_mask]

			seethrough_copy = np.concatenate(([False], seethrough[1:]), 0)

			transp_mask = seethrough_copy[blocks] & (blocks != neighbours[i])
			coords_show_transp = World.coord_array[transp_mask]
			blocks_show_transp = blocks[transp_mask]
			light_show_transp = light_neighb[transp_mask]
			biome_show_transp = chunk_biome[transp_mask]

			transp_mask2 = (neighbours[i] == 8) & (blocks != neighbours[i])
			coords_show_transp2 = World.coord_array[transp_mask2]
			blocks_show_transp2 = neighbours[i][transp_mask2]
			light_show_transp2 = light_neighb[transp_mask2]
			biome_show_transp2 = chunk_biome[transp_mask2]

			coords_show_transp = np.concatenate((coords_show_transp, coords_show_transp2), 0)
			light_show_transp = np.concatenate((light_show_transp, light_show_transp2), 0)
			blocks_show_transp = np.concatenate((blocks_show_transp, blocks_show_transp2), 0)
			biome_show_transp = np.concatenate((biome_show_transp, biome_show_transp2), 0)

			has_biometint = np.repeat(biometint[blocks_show][:, np.newaxis], 2, 1)
			has_biometint_transp = np.repeat(biometint[blocks_show_transp][:, np.newaxis], 2, 1)

			if len(coords_show) > 0:
				c_show_r = np.repeat(coords_show, 6, 0)
				cube_verts = np.tile(Cube.vertices[Cube.triangles[i]], (len(coords_show), 1))

				verts.append(c_show_r + cube_verts - (128, 128, 128))
				tex_verts.append(np.vstack(Textures.game_blocks[blocks_show, 6 * i:6 * i + 6]))
				humtemp = World.get_hum_temp(biome_show, coords_show[:, 1])
				biome_verts.append(np.vstack(np.repeat(
					has_biometint * humtemp - 30000 * ~has_biometint * np.ones(humtemp.shape)
				, 6, 0)))
				normals.append(
				    np.tile(
						types[settings.gpu_data_type][4] * Cube.normals[i], 
						(6 * len(coords_show), 1)
					) * np.tile(
						np.repeat(
							((light_show <= coords_show[:, 1]) + settings.shadow_brightness) /
							(settings.shadow_brightness + 1), 6
						), 
						(3, 1)
					).T
				)

				counter += len(coords_show) * 6
			if len(coords_show_transp) > 0:
				c_show_r = np.repeat(coords_show_transp, 6, 0)
				cube_verts = np.tile(Cube.vertices[Cube.triangles[i]], (len(coords_show_transp), 1))

				transp_verts.append(c_show_r + cube_verts - (128, 128, 128))
				transp_tex_verts.append(np.vstack(Textures.game_blocks[blocks_show_transp, 6 * i:6 * i + 6]))
				humtemp = World.get_hum_temp(biome_show_transp, coords_show_transp[:, 1])
				transp_biome_verts.append(np.vstack(np.repeat(
					has_biometint_transp * humtemp - 30000 * ~has_biometint_transp * np.ones(humtemp.shape),
				 6, 0)))
				transp_normals.append(
					np.tile(
						types[settings.gpu_data_type][4] * Cube.normals[i], 
						(6 * len(coords_show_transp), 1)
					) * np.tile(
						np.repeat(
							((light_show_transp <= coords_show_transp[:, 1]) + settings.shadow_brightness) /
							(settings.shadow_brightness + 1), 6
						), 
						(3, 1)
					).T
				)

				counter_transp += len(coords_show_transp) * 6
		vert_tex_list = np.ravel(
			np.column_stack((
				np.vstack(verts), 
				np.vstack(tex_verts), 
				np.vstack(biome_verts),
				np.vstack(normals)
			))
		).astype(types[settings.gpu_data_type][0])

		if counter_transp != 0:
			vert_tex_transp = np.ravel(
			    np.column_stack((
					np.vstack(transp_verts), 
					np.vstack(transp_tex_verts),
					np.vstack(transp_biome_verts),
					np.vstack(transp_normals)
				))
			).astype(types[settings.gpu_data_type][0])

			return ((vert_tex_list, counter), (vert_tex_transp, counter_transp))
		return ((vert_tex_list, counter), None)

	def chunk_data(coords):
		region, ch = World.get_region(coords)
		if region and ch in region.chunks.keys():
			return region.chunks[ch]
		else:
			chunk = np.zeros((World.chunk_size, World.height, World.chunk_size))
			if World.infinite:
				chunk[:, :World.water_level + 1, :] = 8
				if coords in World.heightmap:
					blockmap = (World.y_array <= World.heightmap[coords])
					chunk[blockmap.transpose(1, 0, 2)] = 3
					World.blockmap[coords] = blockmap
			return chunk

	def get_height(coords):
		chunk = tuple(coords // World.chunk_size)
		local_coords = tuple(coords % World.chunk_size)
		if not chunk in World.heightmap:
			return (settings.heightlim[0] + settings.heightlim[1]) / 2
		return World.heightmap[chunk][local_coords]

	def chunk_in_view(chunk, y_lims):
		left_v = np.array((
			-settings.movement_speed * math.cos(math.radians(player.rot[1] - Display.fovX / 2)),
		    player.norm[1],
		    -settings.movement_speed * math.sin(math.radians(player.rot[1] - Display.fovX / 2))
		))
		right_v = np.array((
			settings.movement_speed * math.cos(math.radians(player.rot[1] + Display.fovX / 2)),
			player.norm[1],
			settings.movement_speed * math.sin(math.radians(player.rot[1] + Display.fovX / 2))
		))
		top_v = np.array((
			player.norm[0],
			settings.movement_speed * abs(math.tan(math.radians(player.rot[0] + 90 + settings.fov_Y))),
			player.norm[2]
		))
		bottom_v = np.array((
			player.norm[0],
			settings.movement_speed * abs(math.tan(math.radians(player.rot[0] - 90 - settings.fov_Y))),
			player.norm[2]
		))
		frustum = (left_v, right_v, top_v, bottom_v)
		in_frustum = True
		for plane in frustum:
			all_inside = False
			for i in range(8):
				a = i >> 2
				b = (i >> 1) & 1
				c = i & 1
				point = (
					np.array((
						chunk [:, 0] + a, 
						y_lims[:, 0] + y_lims[:, 1] * b, 
						chunk [:, 1] + c
					)).T - (
						(player.pos + (0, player.height, 0)) / World.chunk_size
					)
				)
				all_inside |= point @ plane < 0
			in_frustum &= all_inside
		return in_frustum

	def light_data(coords):
		region, ch = World.get_region(coords)
		if region and ch in region.light.keys():
			return region.light[ch]
		else:
			return np.zeros((World.chunk_size, World.chunk_size))

	def get_block(coords):
		if coords is None:
			return None
		if not World.height > coords[1] > 0:
			return 0
		coords = np.array(coords)
		chunk_data = World.chunk_data(tuple(coords[[0, 2]] // World.chunk_size))
		xyz = np.floor(coords).astype(np.int32)
		xyz[[0, 2]] %= World.chunk_size
		return chunk_data[tuple(xyz)]

	def get_biome(coords):
		if coords is None:
			return None
		return World.get_hum_temp(World.biomemap[tuple(coords[[0, 2]] // World.chunk_size)][tuple(coords[[0, 2]] % World.chunk_size)], coords[1])[0] / types[settings.gpu_data_type][4]

	def update_chunk(coords):
		region, ch = World.get_region(coords)
		if not region:
			return
		if ch in region.loaded_chunks.keys():
			glDeleteBuffers(1, int(region.loaded_chunks[ch][0][0]))
			if region.loaded_chunks[ch][1] != None:
				glDeleteBuffers(1, int(region.loaded_chunks[ch][1][0]))
			del region.loaded_chunks[ch], region.preloaded_chunks[ch]
		region.preloaded_chunks[ch] = World.load_chunk(World.process_chunk(coords))

	def set_block(coords, block):
		if coords is None:
			return
		chunk = (coords[0] // World.chunk_size, coords[2] // World.chunk_size)
		region, ch = World.get_region(chunk)
		if not region or coords[1] > World.height:
			print("Cannot build outside world!")
			return
		if block >= len(game_blocks) or block < 0:
			print("Invalid Block!")
			return
		World.put_block(coords, block)
		World.update_chunk(chunk)
		if math.floor(coords[0] % World.chunk_size) == 0:
			World.update_chunk((chunk[0] - 1, chunk[1]))
		elif math.floor(coords[0] % World.chunk_size) == World.chunk_size - 1:
			World.update_chunk((chunk[0] + 1, chunk[1]))
		if math.floor(coords[2] % World.chunk_size) == 0:
			World.update_chunk((chunk[0], chunk[1] - 1))
		elif math.floor(coords[2] % World.chunk_size) == World.chunk_size - 1:
			World.update_chunk((chunk[0], chunk[1] + 1))

	def put_block(coords, block):
		ch = (coords[0] // World.chunk_size, coords[2] // World.chunk_size)
		region, ch = World.get_region(ch)

		current_light = math.floor(region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(
		    coords[2] % World.chunk_size)])
		if block == 0 and World.get_block(coords) != 0 and math.floor(coords[1]) == current_light:
			h = math.floor(coords[1]) - 1
			while World.get_block((coords[0], h, coords[2])) == 0:
				if h < 0:
					break
				h -= 1
			else:
				region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(coords[2] % World.chunk_size)] = h
		elif block != 0 and coords[1] > current_light:
			region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(coords[2] %
			                                                                     World.chunk_size)] = coords[1]
		region.chunks[ch][math.floor(coords[0] % World.chunk_size)][math.floor(coords[1])][math.floor(
		    coords[2] % World.chunk_size)] = block

		World.update_chunk_min_max(coords, block)

	def update_chunk_min_max(coords, block):
		ch = (coords[0] // World.chunk_size, coords[2] // World.chunk_size)
		region, ch = World.get_region(ch)
		if not region or not region.gen_chunks[ch]:
			return
		# Update chunk min and max values
		chmin, chmax = region.chunk_min_max[ch]
		if seethrough[block]:
			chmin_new = min(chmin, (math.floor(coords[1]) - 1) / World.chunk_size)
			chmax_new = region.thorough_chmax(ch) if math.floor(coords[1]) / World.chunk_size >= chmax else chmax
		else:
			chmax_new = max(chmin + chmax, math.floor(coords[1]) / World.chunk_size) - chmin
			chmin_new = region.thorough_chmin(ch) if (math.floor(coords[1]) - 1) / World.chunk_size <= chmin else chmin
		
		# Set some heuristic variables to None to trigger a chunk reload
		if (region.chunk_min_max[ch] != (chmin_new, chmax_new)).any():
			player.old_chunkpos = None
		player.old_rot = None
		
		coords = np.array(coords)
		# Propagate value to neighbouring chunk if on edge
		if ch[0] == 0:
			World.update_chunk_min_max(coords - (World.chunk_size // 2, 0, 0), block)
		elif ch[0] == World.chunk_size - 1:
			World.update_chunk_min_max(coords + (World.chunk_size // 2, 0, 0), block)
		if ch[1] == 0:
			World.update_chunk_min_max(coords - (0, 0, World.chunk_size // 2), block)
		elif ch[1] == World.chunk_size - 1:
			World.update_chunk_min_max(coords + (0, 0, World.chunk_size // 2), block)

		region.chunk_min_max[ch] = (chmin_new, chmax_new)


def make_coord_array():
	global World
	World.coord_array = []
	for i in range(World.chunk_size):
		World.coord_array.append([])
		for j in range(World.height):
			World.coord_array[i].append([])
			for k in range(World.chunk_size):
				World.coord_array[i][j].append((i, j, k))
	World.y_array = np.array(World.coord_array)[:, :, :, 1].transpose(1, 0, 2)
	World.coord_array = np.vstack(World.coord_array)


make_coord_array()
constV = ((0, 0), (0, 0), (0, 0))

game_blocks = None
seethrough = None
biometint = None

character_coords = np.array(((0, 1), (0, 0), (1, 0), (1, 1)))
chat_string = ""


class Display:

	def init(size):
		# Initialising screen, 3D rendering and position/rotation vectors
		if settings.fullscreen:
			if sys.platform == "win32":
				Display.size = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
			else:
				import subprocess
				output = subprocess.Popen('xrandr | grep "*" | cut -d" " -f4', shell=True,
				                          stdout=subprocess.PIPE).communicate()[0]
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
		    settings.render_distance * 3**0.5 * World.chunk_size
		)
		player.rot = np.array((0.0, 0.0, 0.0))
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		glEnableClientState(GL_NORMAL_ARRAY)
		glClearColor(0.0, 0.0, 0.0, 0.0)
		load_shaders()


class Textures:

	def init():
		global game_blocks, seethrough, biometint
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

	def update_pixel_size():
		Textures.pixel_size = 2 * Textures.text[1].get_height() / (Display.size[1] * Textures.texttable_height)

	def generate(blocks):
		Face = np.array([
			[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 2, 2], 
			[0, 1, 1, 1, 2, 3], [0, 1, 2, 2, 3, 4], [0, 1, 2, 3, 4, 5]
		])
		tex_array = []
		for block in blocks:
			cube_sides = np.tile(Cube.sides, (6, 1))
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
		return (clamp(-math.cos((time / settings.day_length) * 2 * math.pi) * hardness, -0.9, 0.9) + 1) / 2


class Time:
	time_start = time.time()
	last_tick = time_start
	last_frame = time_start

	# FPS counting
	last_second = int(time_start)
	frames = 0
	prev_frames = 0


def load_shaders():
	vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/shader.vert", 'r').read(), GL_VERTEX_SHADER)
	fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/shader.frag", 'r').read(), GL_FRAGMENT_SHADER)
	sky_vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/skysh.vert", 'r').read(), GL_VERTEX_SHADER)
	sky_fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/skysh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	water_vertex_shader = compileShader(open(f"shaders/{settings.shader_pack}/watersh.vert", 'r').read(), GL_VERTEX_SHADER)
	#water_fragment_shader = compileShader(open(f"shaders/{settings.shader_pack}/watersh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	global day_night_shader, sky_shader, water_shader
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


def process_chunks(skip_smoothing = False):
	for reg in World.active_regions:
		while (ch := reg.to_be_loaded.pop(0) if len(reg.to_be_loaded) > 0 else None):
			while not skip_smoothing and settings.min_FPS and time.time() - Time.last_frame >= 1 / settings.min_FPS:
				if UI.in_menu:
					return
				time.sleep(0.01)
			reg.preloaded_data[ch] = World.process_chunk(ch + reg.pos)
			World.new_chunks += 1

def chunk_thread():
	try:
		while not UI.in_menu:
			process_chunks()
			time.sleep(1)
	except Exception as e:
		World.thread_exception = e
		print("".join(traceback.format_exc()))

def compute_lighting(blocks):
	not_found = np.full((blocks.shape[0], blocks.shape[2]), True)
	light = np.full((blocks.shape[0], blocks.shape[2]), blocks.shape[1])
	for y in range(blocks.shape[1] - 1, -1, -1):
		not_found &= blocks[:, y, :] == 0
		light[not_found] = y - 1
	return light

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
		if World.get_block(r_pos) not in [0, 8]:
			return (r_pos, o_pos)
		minim = rnd(np.array(r_pos), nm) * invnm
		dt = float(min(minim[minim != 0]))
		o_pos = r_pos + (0, 0, 0)
		r_pos -= dt * 1.1 * player.norm
	return (None, None)


def mode_2D():
	glMatrixMode(GL_PROJECTION)
	glPushMatrix()
	glLoadIdentity()
	glMatrixMode(GL_MODELVIEW)
	glPushMatrix()
	glLoadIdentity()
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)


def mode_3D():
	glPopMatrix()
	glMatrixMode(GL_PROJECTION)
	glPopMatrix()
	glMatrixMode(GL_MODELVIEW)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)


def set_time(time):
	settings.starting_time += time - World.game_time


def clamp(x, bottom, top):
	return max(bottom, min(top, x))


def justify_text(tup):
	x, y = tup
	y = round(y * Display.centre[1]) / Display.centre[1]
	x = round(x * Display.centre[0]) / Display.centre[0]
	return (x, y)


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
Display.init(settings.nominal_res)
Textures.init()
Textures.update_pixel_size()
UI.init_font()
init_schematics()
Sky.init()
Settings.generate_ui()
World.init()
