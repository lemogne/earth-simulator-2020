import pygame as pg
from pygame.locals import *
import math, time, struct, os, sys, json, re, random
from threading import Thread, local
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
from ctypes import *
import settings

# Data Types
BYTE = 0
SHORT = 1
INT = 2
FLOAT = 3
DOUBLE = 4

types = [(np.uint8, GL_BYTE, c_ubyte, 1, 127), (np.uint16, GL_SHORT, c_ushort, 2, 32767),
         (np.uint32, GL_INT, c_uint32, 4, 2147483647), (np.float32, GL_FLOAT, c_float, 4, 1),
         (np.float64, GL_DOUBLE, c_double, 8, 1)]

if settings.gpu_data_type == None:
	size = max(settings.chunk_size, settings.world_height)
	if size <= 256:
		settings.gpu_data_type = BYTE
	elif size <= 32895:
		settings.gpu_data_type = SHORT
	else:
		settings.gpu_data_type = INT

# Menu Options


def quit_game():
	glDisableClientState(GL_VERTEX_ARRAY)
	glDisable(GL_TEXTURE_2D)
	glDisableClientState(GL_TEXTURE_COORD_ARRAY)
	pg.quit()
	quit()


def toggle_menu():
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


	def write(text, loc, size, width=2, color=(1, 1, 1)):
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

		glColor3fv(color)

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

		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glVertexPointer(2, GL_DOUBLE, 0, vert)
		glTexCoordPointer(2, GL_DOUBLE, 0, tex)
		glDrawArrays(GL_QUADS, 0, int(len(tex) / 2))
		glColor3fv((1, 1, 1))

	def input_text(variable, start=0, stop=None):
		if UI.buttons.is_typing():
			variable = variable[:-1]
		else:
			UI.buttons.set_typing(True)
		for event in pg.event.get():
			if event.type == pg.KEYDOWN:
				key = event.key
				if key == 8:
					if len(variable) > start:
						variable = variable[:-1]
					continue
				elif key == 13:
					UI.buttons.set_typing(False)
					return variable
				elif key == 27:
					UI.buttons.set_typing(False)
					return variable[:start]
				if stop == None or len(variable) <= stop:
					variable = variable + event.unicode
			elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
				UI.buttons.set_typing(False)
				UI.buttons.set_input_button(None)
				return variable
		return variable + "_"

	def check_hover(m_pos):
		m_posGL = ((m_pos[0] - Display.centre[0]) / Display.centre[0],
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
				UI.write(button.get_text(),
				         ((box[0] * settings.button_scale + 0.1)[0] * (Display.centre[1] / Display.centre[0]),
				          (box[0] * settings.button_scale - 0.1)[1]),
				         0.05,
				         color=(0, 0, 0))
			else:
				UI.write(button.get_text(),
				         ((box[0] * settings.button_scale + 0.1)[0] * (Display.centre[1] / Display.centre[0]),
				          (box[0] * settings.button_scale - 0.1)[1]), 0.05)

	def type_button(name, var, type_):
		UI.buttons.set_input_button(name)
		UI.buttons[name].set_text(UI.input_text(UI.buttons[name].get_text()))
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

	def run(self):
		if self.function != None:
			self.function()

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

	def type_in(self):
		self.text = UI.input_text(self.text)


def leave_world():
	# Unloads chunks and quits the world
	World.regions = {}
	World.active_regions = {}
	World.loaded_chunks = {}
	World.preloaded_chunks = {}
	UI.paused = False
	UI.in_menu = True
	World.init()


class Start_Game:

	def run():
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
		UI.in_menu = False

	def screen():
		UI.buttons = Start_Game.buttons

	def get_seed():
		UI.buttons.set_input_button("Seed")
		UI.buttons["Seed"].set_text(UI.input_text(UI.buttons["Seed"].get_text()))

	def back():
		global menu_buttons
		UI.buttons = menu_buttons

	buttons = Interface({
	    "WorldSeed": Button((0, 0.5), "World Seed:"),
	    "Seed": Button((0, 0.2), "", True, False, get_seed),
	    "CreateWorld": Button((0, -0.1), "Create World", False, False, run),
	    "Info": Button((0, -0.4), ""),
	    "Back": Button((0, -0.7), "Back", False, False, back)
	})


class Save_World:

	def run():
		global game_blocks
		bytesneeded = lambda x: np.int8(math.log(x, 256) + 1 // 1)

		name = UI.buttons["Name"].get_text()
		try:

			savefile = open(f"worlds/{name}.esw", "wb")  # Create/overwrite world file as given by user input
			savefile.write(b"ES20\x00v0.2\x00")  # Write file header containing game version

			writeBits = lambda data, bits: savefile.write(np.array(data).astype('V' + str(bits)).tobytes())

			# Calculating bytes needed to save world data; saving that information to the file
			BLblocks = bytesneeded(len(game_blocks))
			BLpos = bytesneeded(World.height * (World.chunk_size**2))
			BLch = bytesneeded(2**settings.world_size_F)
			BLheight = bytesneeded(World.height)
			savefile.write(np.array([BLblocks, BLpos, BLch, BLheight, 0], dtype=np.int8).tobytes())
			savefile.write(np.array([World.chunk_size, World.height, 0], dtype=np.int32).tobytes())
			savefile.write(np.array(player.pos, dtype=np.float32).tobytes())
			savefile.write(np.array(player.rot, dtype=np.float32).tobytes())

			# Save each chunk separately, in sequence
			for ch in World.chunks:
				pg.event.get()  # To prevent game window from freezing
				writeBits(ch, BLch)  # Writes the chunk coordinates to the file
				lastblock = None
				counter = 1

				# Block counting loop; compresses block data by making a list of (amount of blocks in a row, block ID); saves to file
				for block in World.chunks[ch].reshape(World.height * (World.chunk_size**2)):
					if block == lastblock:
						counter += 1
					elif lastblock != None:
						writeBits(counter, BLpos)
						writeBits(lastblock, BLblocks)
						counter = 1
						lastblock = block
					else:
						lastblock = block
				writeBits(counter, BLpos)
				writeBits(lastblock, BLblocks)

				# Saves light information
				savefile.write(b"\x00\x00\x00\x00")
				writeBits(World.light[ch].reshape(World.chunk_size**2), BLheight)
				savefile.write(b"\x00\x00\x00\x00")
			savefile.write(b"\x00\x00\x00\x00\x00\x00")

			# Saves important world global variables like the seed or the time
			savefile.write(np.array([World.seed, settings.tree_density], dtype=np.float32).tobytes())
			savefile.write(np.array(World.game_time, dtype=np.int32).tobytes())
			savefile.close()

			UI.buttons["Info"].set_text("World saved successfully!")
		except Exception as e:
			UI.buttons["Info"].set_text(f"Failed to save world: {e}")
		print(UI.buttons["Info"].get_text())

	def screen():
		UI.buttons = Save_World.buttons

	def get_name():
		UI.buttons.set_input_button("Name")
		UI.buttons["Name"].set_text(UI.input_text(UI.buttons["Name"].get_text()))

	def back():
		global paused_buttons
		UI.buttons = paused_buttons

	buttons = Interface({
	    "Name": Button((0, 0.2), "", True, False, get_name),
	    "Save": Button((0, -0.1), "Save World", False, False, run),
	    "Info": Button((0, -0.4), "", None),
	    "Back": Button((0, -0.7), "Back", False, False, back)
	})


class Load_World:

	def run():
		global player, game_blocks, make_coord_array
		# Clear Chunk and Light arrays (dictionaries, whatever)
		World.to_be_loaded = []
		World.preloaded_chunks = {}
		World.preloaded_data = {}
		player.old_chunkpos = None
		World.chunks = {}
		World.light = {}
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
			readfile = open(f"worlds/{name}.esw", "rb")
		except FileNotFoundError:
			UI.buttons["Info"].set_text("No such world!")
			print("No such world!")
			return

		try:
			# read data to variable
			data = readfile.read()
			readfile.close()

			# check if the file is actually a world, and if it is the right version
			if data[:4] != b"ES20":
				UI.buttons["Info"].set_text("This file is not a world!")
				print("This file is not a world!")
				return
			elif data[5:9] not in [b"v0.1", b"v0.2"]:
				UI.buttons["Info"].set_text(f"The version of the world, " + str(data[5:9], "ASCII") +
				                            " is not compatible with this version!")
				print(UI.buttons["Info"].get_text())
				return

			# The world file contains the amount of bytes needed to write certain data (i.e. 1 byte to save a block ID); This data is read here
			BLblocks = data[10]
			BLpos = data[11]
			BLch = data[12]
			BLheight = data[13]
			World.chunk_size = struct.unpack("i", data[15:19])[0]
			World.chunk_min_max = dict()

			# World height, position and camera rotation are read here
			World.height = struct.unpack("i", data[19:23])[0]
			make_coord_array()
			player.pos = np.array((struct.unpack("3f", data[27:39])))
			rv = np.array((struct.unpack("3f", data[39:51])))

			# Chunk reading loop (reads until end of block data flag is read)
			i = 51
			while data[i:i + 6] != b"\x00\x00\x00\x00\x00\x00":
				pg.event.get()  #prevents window from freezing
				ChBuffer = []

				ch = (int.from_bytes(data[i:i + BLch], "little",
				                     signed=True), int.from_bytes(data[i + BLch:i + 2 * BLch], "little",
				                                                  signed=True))  #Chunk position
				i += BLch * 2

				# reads blocks until chunk end flag is read
				while data[i:i + 4] != b"\x00\x00\x00\x00":
					block = int.from_bytes(data[i + BLpos:i + BLpos + BLblocks], "little")
					ChBuffer += [block] * int.from_bytes(data[i:i + BLpos], "little")
					i += BLpos + BLblocks

				# Tries shaping blocks read into a chunk shape; if that is impossible, the the chunk must be malformed and hence the file corrupted
				try:
					World.chunks[ch] = np.reshape(np.array(ChBuffer),
					                              (World.chunk_size, World.height, World.chunk_size)).astype(np.uint8)
				except ValueError:
					UI.buttons["Info"].set_text("World file corrupted!")
					print("World file corrupted!")
					return

				for y in range(World.height):
					if seethrough[World.chunks[ch][:, y, :]].any():
						World.chunk_min_max[ch] = (y / World.chunk_size, 0)
						break
				for y in range(World.height - 1, -1, -1):
					if (World.chunks[ch][:, y, :] != 0).any():
						chmin = World.chunk_min_max[ch][0]
						World.chunk_min_max[ch] = (chmin, (y / World.chunk_size) - chmin)
						break

				i += 4

				# Reads lighting data
				World.light[ch] = np.reshape(
				    np.frombuffer(np.frombuffer(data[i:i + (World.chunk_size**2) * BLheight],
				                                dtype=(f"V{BLheight}")).astype("V4"),
				                  dtype=np.int32), (World.chunk_size, World.chunk_size))
				i += (World.chunk_size**2) * BLheight

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
			settings.tree_density = struct.unpack("f", data[i:i + 4])[0]
			i += 4
			settings.starting_time = struct.unpack("i", data[i:i + 4])[0]

			UI.in_menu = False
			UI.paused = False

			UI.buttons["Info"].set_text("World loaded successfully!")
		except Exception as e:
			UI.buttons["Info"].set_text(f"Failed to load world: {e}")
		print(UI.buttons["Info"].get_text())

	def screen():
		try:
			worldlist = os.listdir("worlds")
			worlds = list(filter(lambda x: (x[-4:] == ".esw"), worldlist))
			Load_World.worlds = [x[:-4] for x in worlds]

		except FileNotFoundError:
			os.mkdir("worlds")
			Load_World.worlds = []
			Load_World.default_buttons["Info"].set_text("Worlds save directory not found!")

		Load_World.pages = []
		for i in range((len(Load_World.worlds) + 3) // 4):
			page = {}
			pl = 4
			if i == len(Load_World.worlds) // 4:
				if len(Load_World.worlds) % 4 != 0:
					pl = len(Load_World.worlds) % 4
			for j in range(pl):
				page[j] = Button((0, 0.4 - j * 0.3), Load_World.worlds[4 * i + j], False, False,
				                 Load_World.gen_func(4 * i + j))
			Load_World.pages.append(Interface(page))
		if len(Load_World.pages) == 0:
			Load_World.pages.append(Interface({}))
		Load_World.reload()
		UI.buttons = Load_World.buttons

	def back():
		global menu_buttons
		UI.buttons = menu_buttons

	def gen_func(world):

		def f():
			for i in range((len(Load_World.worlds) + 3) // 4):
				pl = 4
				if i == len(Load_World.worlds) // 4:
					if len(Load_World.worlds) % 4 != 0:
						pl = len(Load_World.worlds) % 4
				for j in range(pl):
					if 4 * i + j != world:
						Load_World.pages[i][j].texture = Button_Box.type[0]
						Load_World.pages[i][j].input_mode = False
					else:
						Load_World.pages[i][j].texture = Button_Box.type[1]
						Load_World.pages[i][j].input_mode = True
				if Load_World.page == i:
					Load_World.reload()
			Load_World.selected_world = Load_World.worlds[world]

		return f

	def next_page():
		if Load_World.page < len(Load_World.pages) - 1:
			Load_World.page += 1
			Load_World.reload()

	def prev_page():
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
	    "Info": Button((0, -1.15), "", None),
	    "Load": Button((0.5, -0.85), "Load World", False, True, run),
	    "Back": Button((-0.5, -0.85), "Back", False, True, back),
	    "Prev": Button((-1.5, -0.85), "Previous", False, True, prev_page),
	    "Next": Button((1.5, -0.85), "Next", False, True, next_page)
	})
	buttons = default_buttons
	pages = []
	page = 0
	selected_world = None
	worlds = []


class Settings:
	variables = {}

	def main():
		settings.logo_shown = True
		UI.buttons = Settings.buttons

	def cancel():
		global paused_buttons, menu_buttons
		settings.logo_shown = True
		Settings.variables = {}
		if UI.in_menu:
			UI.buttons = menu_buttons
		else:
			UI.buttons = paused_buttons

	def apply():
		setfile = open("settings.py", "r")
		setlines = setfile.readlines()
		setfile.close()
		setfile = open("settings.py", "w")
		for line in setlines:
			var = line.split(" = ")[0]
			if var in Settings.variables:
				comm = line.split("#", 1) + [""]
				setfile.write(f"{var} = {Settings.variables[var]}\t\t{comm[1]}\n")
				exec(f"settings.{var} = {Settings.variables[var]}")
			else:
				setfile.write(line)
		imm = UI.in_menu
		if len(Settings.variables) > 0:
			global unicode, game_blocks, seethrough
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
			init_schematics()
			Sky.init()
			player.init()

			if imm:
				mode_2D()
				World.init()
			else:
				World.load_chunks(True)
				process_chunks()

		Sky.init()
		setfile.close()
		Settings.cancel()

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
			def cat_func():
				UI.buttons = Settings.categories[category]
				settings.logo_shown = False
			return cat_func

		def gen_bool_func(var_name, button_name):
			def button_func():
				Settings.variables[var_name] = str(getattr(settings, var_name) ^ True)
				UI.buttons[button_name].set_text(str(getattr(settings, var_name) ^ True))
			return button_func

		def gen_right_tuple_func(button_name, var_name, var_type):
			def button_func():
				UI.type_button(button_name + "1", var_name, var_type)
			return button_func
		
		def gen_left_tuple_func(button_name, var_name, var_type):
			def button_func():
				UI.type_button(button_name + "0", var_name, var_type)
			return button_func

		def get_button_func(button_name, var_name, var_type):
			return lambda: UI.type_button(button_name, var_name, var_type)

		# Open and read settings file
		setfile = open("settings.py", "r")
		setlines = setfile.readlines()
		setfile.close()

		current_category = ""

		# Button coordinates
		y = -0.9
		x = -0.5
		cat_y = 0.4

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
				button_func = gen_bool_func(var_name, button_name)
			elif var_type is tuple:
				# Match tuple
				tuple_values = re.search(r"\(([\-0-9.]*) *, *([\-0-9.]*)\)", value)

				if not tuple_values:
					# This happens when the tuple has more than two values or something is malformed
					# This is complicated to handle and will be left out for now
					continue

				# Special case: right value of 2-tuple
				right_val = tuple_values.group(2)
				right_button_func = gen_right_tuple_func(button_name, var_name, var_type)
				Settings.categories[current_category].buttons[button_name + "1"] = Button((x, y), right_val, True, True, right_button_func)

				# left value adjusted so that rest of function can carry on as normal
				button_func = gen_left_tuple_func(button_name, var_name, var_type)
				button_name += "0"
				y += 0.25
				value = tuple_values.group(1)
			else:
				button_func = get_button_func(button_name, var_name, var_type)

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
    "Main": Button((0, -0.8), "Back to Main Menu", False, False, leave_world),
    "Save": Button((0, -0.4), "Save World", False, False, Save_World.screen),
    "Settings": Button((0, 0), "Settings", False, False, Settings.main),
    "Resume": Button((0, 0.4), "Resume Game", False, False, toggle_menu)
})

menu_buttons = Interface({
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
				forward /= 1.41421356237
				sideways /= 1.41421356237
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

			# Check for block collisions
			segments = math.ceil(np.linalg.norm(self.mv * dt))
			for j in range(segments):
				for i in range(3):
					if self.check_in_block(i, dt / segments, self.mv, self.pos):
						if i == 1:
							offset = 0 if self.mv[i] > 0 else (1 - (settings.player_height % 1))
						else:
							offset = settings.player_width
						offset += settings.hitbox_epsilon
						if self.mv[i] < 0:
							self.pos[i] = math.ceil(self.pos[i]) - offset
						elif self.mv[i] > 0:
							self.pos[i] = math.floor(self.pos[i]) + offset
						self.mv[i] = 0
				self.pos -= self.mv * (dt / segments)

			# MOVEMENT
			self.chunkpos = self.pos // World.chunk_size

	def check_in_block(self, dim, dt, mv, pos):
		# TODO: optimise!!!
		hitbox_min = pos - (settings.player_width, 0, settings.player_width)
		hitbox_max = pos + (settings.player_width, settings.player_height, settings.player_width)

		if mv[dim] > 0:
			hitbox_min[dim] -= mv[dim] * dt
			hitbox_max[dim] = hitbox_min[dim]
		else:
			hitbox_max[dim] -= mv[dim] * dt
			hitbox_min[dim] = hitbox_max[dim]

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
				rv = np.array((max(min((m_y * settings.mouse_sensitivity), 90 - self.rot[0]), -90 - self.rot[0]),
				            (m_x * settings.mouse_sensitivity), 0))
				self.rot += rv
				pg.mouse.set_pos(Display.centre)
			else:
				rv = np.array((0.0, 0.0, 0.0))

			# norm is the normal vector of the culling plane, also the 'forward' vector
			self.norm = np.array((-settings.movement_speed * math.sin(math.radians(self.rot[1])),
			                   -settings.movement_speed * math.tan(math.radians(self.rot[0])),
			                   settings.movement_speed * math.cos(math.radians(self.rot[1]))))

			glRotatef(rv[1], 0, 1, 0)
			glRotatef(rv[0], -self.norm[2], 0, self.norm[0])


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
	pos = (0, 0)
	chunk_coords = None
	in_view = None
	chunk_y_lims = None
	lock = True
	gen_chunks = None

	def __init__(self, pos):
		self.preloaded_chunks = dict()
		self.loaded_chunks = dict()
		self.to_be_loaded = list()
		self.preloaded_data = dict()
		self.chunk_min_max = np.full((settings.region_size, settings.region_size, 2), 0.0)
		self.chunks = dict()
		self.light = dict()
		self.pos = np.array(pos) * settings.region_size
		self.chunk_coords = None
		self.lock = False
		self.gen_chunks = np.full((settings.region_size, settings.region_size), False)


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

	def load_chunks(self, change_pos, change_rot, ForceLoad=False):
		if self.lock:
			return
		if change_pos or self.chunk_coords is None or ForceLoad:
			self.chunk_coords = np.mgrid[0:settings.region_size, 0:settings.region_size].T[:, :, ::-1]
			chunk_distance = settings.chunk_distance(abs(self.chunk_coords[:, :, 0] - player.chunkpos[0] + self.pos[0]),
			                                         abs(self.chunk_coords[:, :, 1] - player.chunkpos[2] + self.pos[1]))
			self.chunk_coords = self.chunk_coords[chunk_distance <= settings.render_distance]
			gen_status = self.gen_chunks[chunk_distance <= settings.render_distance]
			to_generate = self.chunk_coords[~gen_status]
			self.chunk_coords = self.chunk_coords[gen_status]
			self.chunk_y_lims = self.chunk_min_max[chunk_distance <= settings.render_distance][gen_status]
			player.old_rot = None

			for ch in to_generate:
				key = tuple((ch + self.pos).astype(np.int32))
				if key not in World.chunks_to_generate:
					World.chunks_to_generate.append(key)
		if ForceLoad:
			self.in_view = np.full(shape=len(self.chunk_coords), fill_value=True)
			player.old_rot = None
		elif change_pos or change_rot:
			self.in_view = World.chunk_in_view(self.chunk_coords + self.pos, self.chunk_y_lims)
		else:
			return
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
	heightmap = {}
	trees = {}
	regions = {}
	chunks_to_generate = []
	active_regions = []

	def init():
		World.height = settings.world_height
		World.chunk_size = settings.chunk_size
		World.game_time = settings.starting_time

	def load_chunks(ForceLoad=False):
		reg_max = (player.chunkpos + settings.render_distance) // settings.region_size
		reg_min = (player.chunkpos - settings.render_distance) // settings.region_size
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
					World.regions[(i, j)] = Region((i, j))
				World.regions[(i, j)].load_chunks(change_pos, change_rot, ForceLoad)
				World.active_regions.append(World.regions[(i, j)])

	def load_chunk(chunkdata):
		vert_tex_list = chunkdata[0][0]
		counter = chunkdata[0][1]
		vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo)
		glBufferData(GL_ARRAY_BUFFER,
		             len(vert_tex_list) * types[settings.gpu_data_type][3],
		             (types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list), GL_STATIC_DRAW)

		if chunkdata[1] != None:
			vert_tex_list = chunkdata[1][0]
			counter_transp = chunkdata[1][1]
			vbo_transp = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, vbo_transp)
			glBufferData(GL_ARRAY_BUFFER,
			             len(vert_tex_list) * types[settings.gpu_data_type][3],
			             (types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list), GL_STATIC_DRAW)
			return ((vbo, counter), (vbo_transp, counter_transp))
		return ((vbo, counter), None)

	def get_region(chunkpos):
		reg_coords = (chunkpos[0] // settings.region_size, chunkpos[1] // settings.region_size)
		if reg_coords in World.regions:
			reg = World.regions[reg_coords]
		else:
			reg = None
		ch = (int(chunkpos[0] % settings.region_size), int(chunkpos[1] % settings.region_size))
		return (reg, ch)

	def process_chunk(chunkpos):
		region, ch = World.get_region(chunkpos)
		chunk = region.chunks[ch]
		blocks = np.vstack(chunk)
		chunk_light = region.light[ch]
		
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
		normals = []
		transp_verts = []
		transp_tex_verts = []
		transp_normals = []
		counter = 0
		counter_transp = 0
		for i in range(6):
			# Basically, for each of the 6 possible faces of all cubes, we filter out all those, whose neighbour is not air;
			# the remainder we turn into vertex and texture data
			neighbours[i] = np.vstack(neighbours[i]).astype(int)
			lnb = np.vstack(
			    np.reshape(np.tile(neighbours_light[i], World.height),
			               (World.chunk_size, World.height, World.chunk_size)))
			nbWater = neighbours[i] == 8
			nb_transp = seethrough[neighbours[i]]
			bWater = blocks[nbWater]
			cWater = coordArray[nbWater]
			lWater = lnb[nbWater]

			b_transp = blocks[nb_transp]
			c_transp = coordArray[nb_transp]
			l_transp = lnb[nb_transp]

			SMask = ~seethrough[b_transp]
			bShow = b_transp[SMask]  # Solid blocks
			cShow = c_transp[SMask]
			lShow = l_transp[SMask]

			TMask1 = seethrough[blocks] & (blocks != neighbours[i])
			cShow_transp = coordArray[TMask1]
			bShow_transp = blocks[TMask1]
			lShow_transp = lnb[TMask1]

			# Water blocks, also air blocks bordering on water (so that you can see water surface from below)
			cShow_transp = np.concatenate((cShow_transp[bShow_transp != 0], cWater[bWater == 0]), 0)
			lShow_transp = np.concatenate((lShow_transp[bShow_transp != 0], lWater[bWater == 0]), 0)
			bShow_transp = np.concatenate((bShow_transp[bShow_transp != 0], bWater[bWater == 0]), 0)

			if len(cShow) > 0:
				cShowR = np.repeat(cShow, 6, 0)
				cube_verts = np.tile(Cube.vertices[Cube.triangles[i]], (len(cShow), 1))

				verts.append(cShowR + cube_verts - (128, 128, 128))
				tex_verts.append(np.vstack(Textures.game_blocks[bShow, 6 * i:6 * i + 6]))
				normals.append(
				    np.tile(types[settings.gpu_data_type][4] * Cube.normals[i], (6 * len(cShow), 1)) * np.tile(
				        np.repeat(((lShow <= cShow[:, 1]) + settings.shadow_brightness) /
				                  (settings.shadow_brightness + 1), 6), (3, 1)).T)

				counter += len(cShow) * 6
			if len(cShow_transp) > 0:
				cShowR = np.repeat(cShow_transp, 6, 0)
				cube_verts = np.tile(Cube.vertices[Cube.triangles[i]], (len(cShow_transp), 1))

				transp_verts.append(cShowR + cube_verts - (128, 128, 128))
				transp_tex_verts.append(np.vstack(Textures.game_blocks[bShow_transp, 6 * i:6 * i + 6]))
				transp_normals.append(
				    np.tile(types[settings.gpu_data_type][4] * Cube.normals[i], (6 * len(cShow_transp), 1)) * np.tile(
				        np.repeat(((lShow_transp <= cShow_transp[:, 1]) + settings.shadow_brightness) /
				                  (settings.shadow_brightness + 1), 6), (3, 1)).T)

				counter_transp += len(cShow_transp) * 6
		vert_tex_list = np.ravel(np.column_stack(
		    (np.vstack(verts), np.vstack(tex_verts), np.vstack(normals)))).astype(types[settings.gpu_data_type][0])
		if counter_transp != 0:
			vert_tex_transp = np.ravel(
			    np.column_stack((np.vstack(transp_verts), np.vstack(transp_tex_verts),
			                     np.vstack(transp_normals)))).astype(types[settings.gpu_data_type][0])
			return ((vert_tex_list, counter), (vert_tex_transp, counter_transp))
		return ((vert_tex_list, counter), None)

	def chunk_data(coords):
		region, ch = World.get_region(coords)
		if region and ch in region.chunks.keys():
			return region.chunks[ch]
		else:
			return np.zeros((World.chunk_size, World.height, World.chunk_size))

	def chunk_in_view(chunk, y_lims):
		leftV = np.array((-settings.movement_speed * math.cos(math.radians(player.rot[1] - Display.fovX / 2)),
		               player.norm[1],
		               -settings.movement_speed * math.sin(math.radians(player.rot[1] - Display.fovX / 2))))
		rightV = np.array((settings.movement_speed * math.cos(math.radians(player.rot[1] + Display.fovX / 2)),
		                player.norm[1],
		                settings.movement_speed * math.sin(math.radians(player.rot[1] + Display.fovX / 2))))
		topV = np.array((player.norm[0],
		              settings.movement_speed * abs(math.tan(math.radians(player.rot[0] + 90 + settings.fov_Y))),
		              player.norm[2]))
		bottomV = np.array((player.norm[0],
		                 settings.movement_speed * abs(math.tan(math.radians(player.rot[0] - 90 - settings.fov_Y))),
		                 player.norm[2]))
		frustum = (leftV, rightV, topV, bottomV)
		inFrustum = True
		for plane in frustum:
			all_inside = False
			for i in range(8):
				a = i >> 2
				b = (i >> 1) & 1
				c = i & 1
				point = (np.array((chunk[:, 0] + a, y_lims[:, 0] + y_lims[:, 1] * b, chunk[:, 1] + c)).T -
				         ((player.pos + (0, player.height, 0)) / World.chunk_size))
				all_inside |= point @ plane < 0
			inFrustum &= all_inside
		return inFrustum

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
		return World.chunk_data((coords[0] // World.chunk_size, coords[2] // World.chunk_size))[math.floor(
		    coords[0] % World.chunk_size)][math.floor(coords[1])][math.floor(coords[2] % World.chunk_size)]

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

		currentLight = math.floor(region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(
		    coords[2] % World.chunk_size)])
		if block == 0 and World.get_block(coords) != 0 and math.floor(coords[1]) == currentLight:
			h = math.floor(coords[1]) - 1
			while World.get_block((coords[0], h, coords[2])) == 0:
				if h < 0:
					break
				h -= 1
			else:
				region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(coords[2] % World.chunk_size)] = h
		elif block != 0 and coords[1] > currentLight:
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

coordArray = []


def make_coord_array():
	global coordArray, coordArray3
	coordArray = []
	for i in range(World.chunk_size):
		coordArray.append([])
		for j in range(World.height):
			coordArray[i].append([])
			for k in range(World.chunk_size):
				coordArray[i][j].append((i, j, k))
	coordArray3 = np.array(coordArray)
	coordArray = np.vstack(coordArray)


make_coord_array()
constV = ((0, 0), (0, 0), (0, 0))

game_blocks = None
seethrough = None

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
				output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True,
				                          stdout=subprocess.PIPE).communicate()[0]
				output = output.decode()[:-1].split("x")
				Display.size = (int(output[0]), int(output[1]))
		else:
			Display.size = size
		Display.fovX = math.degrees(
		    2 * math.atan(math.tan(math.radians(settings.fov_Y / 2)) * (Display.size[0] / Display.size[1])))
		Display.centre = (Display.size[0] // 2, Display.size[1] // 2)

		Display.screen = pg.display.set_mode(Display.size, (FULLSCREEN * settings.fullscreen) | DOUBLEBUF | OPENGL
		                                     | (pg.RESIZABLE * settings.resizeable))
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		gluPerspective(settings.fov_Y, (Display.size[0] / Display.size[1]), 0.1,
		               settings.render_distance * 3**0.5 * World.chunk_size)
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
		global game_blocks, seethrough
		game_blocks = json.loads(open(f"textures/{settings.texture_pack}/texturing.json").read())
		seethrough = np.array(json.loads(open(f"textures/{settings.texture_pack}/transparency.json").read()))
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

		title_size = (Textures.title[1].get_width(), Textures.title[1].get_height())
		Textures.text_ratio = Textures.text[1].get_width() * (Textures.texttable_height /
		                                                      16) / Textures.text[1].get_height()
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
		Face = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 3],
		                 [0, 1, 2, 2, 3, 4], [0, 1, 2, 3, 4, 5]])
		TexArray = []
		for block in blocks:
			CS = np.tile(Cube.sides, (6, 1))
			tiles = np.array(block)[Face[len(block) - 1]]
			BR = np.repeat(
			    np.array([tiles % Textures.mapsize[0], -(tiles + 1) // Textures.mapsize[0]]).T, 6, 0)
			TexArray.append(CS + BR)
		return np.array(TexArray)

	def load(file, filter = GL_NEAREST):
		textureSurface = pg.image.load(f"textures/{settings.texture_pack}/{file}")
		textureData = pg.image.tostring(textureSurface, "RGBA", 1)

		glEnable(GL_TEXTURE_2D)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		texid = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, texid)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureSurface.get_width(), textureSurface.get_height(), 0, GL_RGBA,
					GL_UNSIGNED_BYTE, textureData)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter)
		return (texid, textureSurface)

class Sky:
	triangles = ((2, 0, 5), (0, 3, 5), (1, 2, 5), (3, 1, 5), (6, 8, 4), (9, 6, 4), (8, 7, 4), (7, 9, 4), (6, 2, 8),
	             (6, 0, 2), (9, 0, 6), (9, 3, 0), (8, 1, 7), (8, 2, 1), (7, 3, 9), (7, 1, 3))
	vertices = ((-1, 1, -1), (1, 1, 1), (-1, 1, 1), (1, 1, -1), (0, -1, 0), (0, 4 / 3, 0), (-1, 0, -1), (1, 0, 1),
	            (-1, 0, 1), (1, 0, -1))
	vert_list = np.array(vertices)[np.array(triangles).ravel()]
	normals = np.zeros(len(vert_list))

	def init():
		t = 0.05
		b = 0.95
		m = (2 * t + b) / 3
		Sky.tex_list = np.array([(0, m), (0, m), (0, t)] * 4 + [(0, b), (0, b), (0, b)] * 4 +
		                        [(0, b), (0, m), (0, b), (0, b), (0, m), (0, m)] * 4)

	def get_tex():
		tempS = Sky.texture_offset(World.game_time)
		return Sky.tex_list + (tempS, 0)

	def texture_offset(time):
		hardness = 2
		return (clamp(-math.cos((time / settings.day_length) * 2 * math.pi) * hardness, -0.9, 0.9) + 1) / 2



def load_shaders():
	vertexShader = compileShader(open(f"shaders/{settings.shader_pack}/shader.vert", 'r').read(), GL_VERTEX_SHADER)
	fragmentShader = compileShader(open(f"shaders/{settings.shader_pack}/shader.frag", 'r').read(), GL_FRAGMENT_SHADER)
	skyVertSh = compileShader(open(f"shaders/{settings.shader_pack}/skysh.vert", 'r').read(), GL_VERTEX_SHADER)
	skyFragSh = compileShader(open(f"shaders/{settings.shader_pack}/skysh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	waterVertSh = compileShader(open(f"shaders/{settings.shader_pack}/watersh.vert", 'r').read(), GL_VERTEX_SHADER)
	#waterFragSh= compileShader(open(f"shaders/{settings.shader_pack}/watersh.frag", 'r').read(), GL_FRAGMENT_SHADER)
	global DayNightShader, skyShader, waterShader
	DayNightShader = glCreateProgram()
	glAttachShader(DayNightShader, vertexShader)
	glAttachShader(DayNightShader, fragmentShader)
	glLinkProgram(DayNightShader)

	skyShader = glCreateProgram()
	glAttachShader(skyShader, skyVertSh)
	glAttachShader(skyShader, skyFragSh)
	glLinkProgram(skyShader)

	waterShader = glCreateProgram()
	glAttachShader(waterShader, waterVertSh)
	glAttachShader(waterShader, fragmentShader)
	glLinkProgram(waterShader)


def process_chunks():
	for reg in World.active_regions:
		if reg.lock:
			continue
		while (ch := reg.to_be_loaded.pop(0) if len(reg.to_be_loaded) > 0 else None):
			reg.preloaded_data[ch] = World.process_chunk(ch + reg.pos)

def chunk_thread():
	try:
		while not UI.in_menu:
			process_chunks()
			time.sleep(1)
	except Exception as e:
		World.thread_exception = e
		print(e)

def schematic_lighting(schematic):
	not_found = np.full((schematic.shape[0], schematic.shape[2]), True)
	light = np.full((schematic.shape[0], schematic.shape[2]), schematic.shape[1])
	for y in range(schematic.shape[1] - 1, -1, -1):
		not_found &= schematic[:, y, :] == 0
		light[not_found] = y - 1
	return light

def get_schematic(file):
	raw_json = open(f"schematics/{settings.schematic_pack}/{file}.json").read()
	schem = np.array(json.loads(raw_json))
	return (schem, schematic_lighting(schem))


def get_looked_at():

	def rnd(p, dx):
		return (dx < 0) * (p // 1) - (dx > 0) * ((-p) // 1) - p

	rPos = player.pos + (0, player.height, 0)
	oPos = rPos
	dt = 0
	nm = np.array(-player.norm)
	invnm = 1 / nm
	while np.linalg.norm(rPos - (player.pos + (0, player.height, 0))) <= settings.hand_reach:
		if World.get_block(rPos) not in [0, 8]:
			return (rPos, oPos)
		minim = rnd(np.array(rPos), nm) * invnm
		dt = float(min(minim[minim != 0]))
		oPos = rPos + (0, 0, 0)
		rPos -= dt * 1.1 * player.norm
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
