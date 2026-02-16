#from init import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
import numpy as np
import math, time, struct, os, sys, json, re, traceback
import settings
from textures import *
from world import *
import render_blocks
import opensimplex as noise


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
		[ # Long Button
			((0, 1), (1, 1), (1, 0.875), (0, 0.875)),
			((0, 0.75), (1, 0.75), (1, 0.625), (0, 0.625))
		],
		[ # Long Textbox
			((0, 0.625), (1, 0.625), (1, 0.5), (0, 0.5)),
			((0, 0.375), (1, 0.375), (1, 0.25), (0, 0.25))
		],
		[ # Short Button
			((0, 0.875), (0.5, 0.875), (0.5, 0.75), (0, 0.75)),
			((0.5, 0.875), (1, 0.875), (1, 0.75), (0.5, 0.75))
		],
		[ # Short Textbox
			((0, 0.5), (0.5, 0.5), (0.5, 0.375), (0, 0.375)),
			((0.5, 0.5), (1, 0.5), (1, 0.375), (0.5, 0.375))
		],
		[ # Tiny Button
			((0, 0.25), (0.25, 0.25), (0.25, 0.125), (0, 0.125)),
			((0.25, 0.25), (0.5, 0.25), (0.5, 0.125), (0.25, 0.125))
		],
		[ # Tiny Textbox
			((0.5, 0.25), (0.75, 0.25), (0.75, 0.125), (0.5, 0.125)),
			((0.75, 0.25), (1.0, 0.25), (1.0, 0.125), (0.75, 0.125))
		],
	])
	vertices_long = np.array(((-1, 0.125), (1, 0.125), (1, -0.125), (-1, -0.125)))
	vertices_short = np.array(((-0.5, 0.125), (0.5, 0.125), (0.5, -0.125), (-0.5, -0.125)))
	vertices_tiny = np.array(((-0.5, 0.125), (0, 0.125), (0, -0.125), (-0.5, -0.125)))


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
	info_text = None
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


	def print_info(text):
		UI.info_text.add_text(text)
		print(text)


	def clear_info():
		UI.info_text.set_text('')



class Button:
	def __init__(self, position, text, input_mode=None, size=0, function=None):
		if input_mode == None:
			self.texture = None
		else:
			if size == None:
				size = 0
			self.texture = Button_Box.type[2 * size + input_mode]

		if size == 2:
			self.box = Button_Box.vertices_tiny
		elif size == 1:
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

	def add_text(self, text):
		self.text += '\n' + text

	def is_text_box(self):
		return self.input_mode

	def type_in(self, event):
		self.text = UI.input_text(self.text, event)


UI.info_text = Button((-0.8, -1.2), "")


def leave_world(event):
	# Unloads chunks and quits the world
	save_world(event)
	World.regions = {}
	World.active_regions = {}
	World.loaded_chunks = {}
	World.preloaded_chunks = {}
	UI.paused = False
	UI.in_menu = True
	World.init(World.game_blocks, World.seethrough, World.biome_tint)


class Start_Game:

	def run(event):
		UI.clear_info()
		World.new = True
		# Determining world seed:
		# if field is empty, generate random seed; else, try to interpret seed as number. If not possible, display error.
		if UI.buttons["Seed"].get_text() == "":
			World.seed = int((time.time() % 1) * 10000000)
		else:
			try:
				World.seed = int(UI.buttons["Seed"].get_text())
			except ValueError:
				UI.print_info("World seed must be an integer!")
				return

		worldname = UI.buttons["Name"].get_text()
		if worldname == "":
			worldname = "world_" + time.strftime("%Y-%m-%d_%H%M%S")
		filepath = f"worlds/{worldname}.esw"
		if os.path.exists(filepath):
			UI.print_info(f"World '{worldname}' already exists!")
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
	    "WorldName": Button((-1.5, 0.5), "World Name:", None, 1),
	    "Name": Button((0, 0.5), "", True, 0, get_name),
	    "WorldSeed": Button((-1.5, 0.2), "World Seed:", None, 1),
	    "Seed": Button((0, 0.2), "", True, 0, get_seed),
	    "CreateWorld": Button((0, -0.2), "Create World", False, 0, run),
	    "Info": UI.info_text,
	    "Back": Button((0, -0.7), "Back", False, 0, back)
	})



def save_world(event):
	try:
		raw_region_data = {}
		sorted_reg_table = list(World.region_table.keys())
		sorted_reg_table.sort(key = lambda x: World.region_table[x][0])
		savefile = World.file
		savefile.seek(0, 2)
		filesize = savefile.tell()
		for i, reg in enumerate(sorted_reg_table):
			if reg not in World.regions:
				next_reg = World.region_table[sorted_reg_table[i + 1]][0] if i + 1 < len(sorted_reg_table) else filesize
				raw_region_data[reg] = Load_World.load_region_raw(reg, next_reg)

		savefile.seek(0, 0)
		savefile.truncate(0)
		savefile.write(b"ES20\x00v0.4")  # Write file header containing game version

		write_bytes = lambda data, bytes: savefile.write(np.array(data).astype(f'V{bytes}').tobytes())

		# Calculating bytes needed to save world data; saving that information to the file
		BLblocks = bytesneeded(len(World.game_blocks))

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


		UI.print_info("World saved successfully!")
	except Exception as e:
		UI.print_info(f"Failed to save world: {repr(e)}")


class Load_World:

	def run(event):
		global player, make_coord_array
		# Clear Chunk and Light arrays (dictionaries, whatever)
		player.old_chunkpos = None
		World.regions = {}
		World.active_regions = {}
		World.regions_to_load = []
		World.new = False

		UI.clear_info()

		# Check if a world has been selected
		if Load_World.selected_world:
			name = Load_World.selected_world
		else:
			UI.print_info("No world selected!")
			return

		# Check if world actually exists and open it
		try:
			readfile = open(f"worlds/{name}.esw", "r+b")
		except FileNotFoundError:
			UI.print_info("No such world!")
			return

		try:
			UI.buttons = loading_buttons
			# check if the file is actually a world, and if it is the right version
			magic_const = readfile.read(4)
			readfile.seek(1, 1)
			version = readfile.read(4)
			if magic_const != b"ES20":
				UI.print_info("This file is not a world!")
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
				World.starting_time = struct.unpack("i", readfile.read(4))[0]
				World.game_time = World.starting_time

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
				UI.print_info(
					f"This world is from version {str(version, 'ASCII')} and uses an old format.\n"\
					"Please convert it by re-saving it in this version.\n"\
					"Worlds in this format will not be supported much longer."
				)

				World.bytes_for_block_ID = struct.unpack("b", readfile.read(1))[0]
				World.infinite = struct.unpack("b", readfile.read(1))[0] == 0
				World.region_size = struct.unpack("b", readfile.read(1))[0]
				World.height = struct.unpack("i", readfile.read(4))[0]
				World.chunk_size = struct.unpack("I", readfile.read(4))[0]

				make_coord_array()
				player.pos = np.array((struct.unpack("3f", readfile.read(12))))
				player.rot = np.array((struct.unpack("3f", readfile.read(12))))
				World.starting_time = struct.unpack("i", readfile.read(4))[0]

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
				UI.print_info(
					f"This world is from version {str(version, 'ASCII')} and uses an old format.\n"\
					"Please convert it by re-saving it in this version.\n"\
					"Worlds in this format will not be supported much longer."
				)

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
						UI.print_info("World file corrupted!")
						return

					for y in range(World.height):
						if World.seethrough[region.chunks[ch][:, y, :]].any():
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

					# Generate biome data
					World.gen_biomemap(chunk, np.arange(World.chunk_size), np.arange(World.chunk_size))

					# Check if chunk end flag is present; if not, the file must be corrupted
					if data[i:i + 4] != b"\x00\x00\x00\x00":
						UI.print_info("World file corrupted!")
						return
					i += 4
				i += 6

				# Read important world information
				World.seed = struct.unpack("f", data[i:i + 4])[0]
				i += 4
				settings.tree_density_mean = struct.unpack("f", data[i:i + 4])[0]
				i += 4
				World.starting_time = struct.unpack("i", data[i:i + 4])[0]
			else:
				UI.print_info(f"The version of the world, {str(version, 'ASCII')} is not compatible with this version!")
				return
			UI.in_menu = False
			UI.paused = False

			UI.print_info("World loaded successfully!")
		except Exception as e:
			UI.print_info(f"Failed to load world: {'\n'.join(traceback.format_exc())}")


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
				if World.seethrough[region.chunks[ch][:, y, :]].any():
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


	def load_region_raw(reg, next_reg_location):
		readfile = World.file
		readfile.seek(World.region_table[reg][0])
		region_buffer = readfile.read(next_reg_location - World.region_table[reg][0])
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
			UI.print_info.set_text("Worlds save directory not found!")

		Load_World.pages = []
		for i in range((len(Load_World.worlds) + 3) // settings.worlds_per_page):
			page = {}
			page_length = settings.worlds_per_page
			if i == len(Load_World.worlds) // settings.worlds_per_page:
				if len(Load_World.worlds) % settings.worlds_per_page != 0:
					page_length = len(Load_World.worlds) % settings.worlds_per_page
			for j in range(page_length):
				page[j] = Button((0, 0.4 - j * 0.3), Load_World.worlds[settings.worlds_per_page * i + j], False, 0,
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
	    "Title": Button((-0.5, 0.7), "Worlds", None, 1),
	    "Page": Button((0.5, 0.7), "Page 1", None, 1),
	    "Info": UI.info_text,
	    "Load": Button((0.5, 0.4 - settings.worlds_per_page * 0.3), "Load World", False, 1, run),
	    "Back": Button((-0.5, 0.4 - settings.worlds_per_page * 0.3), "Back", False, 1, back),
	    "Prev": Button((-1.5, 0.4 - settings.worlds_per_page * 0.3), "Previous", False, 1, prev_page),
	    "Next": Button((1.5, 0.4 - settings.worlds_per_page * 0.3), "Next", False, 1, next_page)
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
			global unicode
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
					render_blocks.mode_2D()
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
					Settings.buttons.buttons[current_category] = Button((0, cat_y), current_category, False, 0, gen_cat_func(current_category))
					Settings.categories[current_category] = Interface({"Back": Button((0.67, -1.2), "Back", False, 1, Settings.main)})
					cat_y -= 0.3
					y = -0.9
					x = -1
				continue

			# Skip hidden settings
			if comment and comment[:6] == "hidden":
				continue

			button_name = var_name + "_value"
			var_type = datatype(value)
			button_size = 1

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
				Settings.categories[current_category].buttons[button_name + "1"] = Button((x + 0.5, y), right_val, True, 2, right_button_func)

				# left value adjusted so that rest of function can carry on as normal
				button_func = gen_left_tuple_func(button_name, var_name, var_type, current_category == "Graphics")
				button_name += "0"
				#y += 0.25
				value = tuple_values.group(1)
				button_size = 2
			else:
				button_func = get_button_func(button_name, var_name, var_type, current_category == "Graphics")

			# If string, extract value
			if var_type is str:
				value = re.sub("[\"'](.*)[\"']", r"\1", value)

			# Insert new setting into correct category
			Settings.categories[current_category].buttons[var_name+"_label"] = Button((x - 1, y), gen_name(var_name), None, 1)
			Settings.categories[current_category].buttons[button_name] = Button((x, y), value, var_type != bool, button_size, button_func)

			# Coordinate calculations
			y += 0.3
			if y > 1.2:
				y = -0.9
				x = 1.5

	buttons = Interface({
	    "Cancel": Button((0.67, -1.2), "Cancel", False, 1, cancel),
	    "OK": Button((-0.67, -1.2), "OK", False, 1, apply)
	})

	categories = {}


paused_buttons = Interface({
	"Info": UI.info_text,
    "Main": Button((0, -0.8), "Back to Main Menu", False, 0, leave_world),
    "Save": Button((0, -0.4), "Save World", False, 0, save_world),
    "Settings": Button((0, 0), "Settings", False, 0, Settings.main),
    "Resume": Button((0, 0.4), "Resume Game", False, 0, toggle_menu)
})

menu_buttons = Interface({
	"Info": UI.info_text,
    "Quit": Button((0, -0.9), "Quit Game", False, 0, quit_game),
    "Settings": Button((0, -0.5), "Settings", False, 0, Settings.main),
    "Load": Button((0, -0.1), "Load World", False, 0, Load_World.screen),
    "New": Button((0, 0.3), "New World", False, 0, Start_Game.screen)
})


loading_buttons = Interface({
	"Info": UI.info_text
})


#MAIN MENU
def run():
	glDisableClientState(GL_NORMAL_ARRAY)
	UI.buttons = menu_buttons
	glEnable(GL_BLEND)
	render_blocks.mode_2D()
	now = time.time()
	last_frame = now

	#UI.clear_info()

	while UI.in_menu:
		now = time.time()
		if not settings.frame_cap or settings.max_FPS * (now - last_frame) >= 1:
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			glBegin(GL_QUADS)
			for i in range(4):
				glTexCoord2fv(Textures.title_coords[i])
				glVertex2fv(character_coords[i] * 2 - 1)
			glEnd()

			if settings.logo_shown:
				glBindTexture(GL_TEXTURE_2D, Textures.logo[0])
				glBegin(GL_QUADS)
				for i in range(4):
					glTexCoord2fv(character_coords[i])
					glVertex2fv(Textures.logo_coords[i])
				glEnd()

			UI.render_buttons()
			UI.write("v0.4", (0.875, -0.98), 0.05, color=(1, 1, 1), shadow=True)
			glBindTexture(GL_TEXTURE_2D, Textures.title[0])
			pg.display.flip()

		UI.check_hover(pg.mouse.get_pos())

		for event in pg.event.get():
			if UI.buttons.is_typing():
				if UI.buttons.get_input_button() == None:
					chat_string = UI.input_text(chat_string, event, start=2)
					if not UI.buttons.is_typing():
						try:
							exec(chat_string[2:])
						except Exception:
							chat_string = "Invalid Python statement!"
							print("Invalid Python statement!")
				else:
					UI.buttons.get_input_button().run(event)
					if not UI.buttons.is_typing():
						UI.buttons.set_typing(None)
			if event.type == pg.QUIT:
				quit_game()
			elif event.type == pg.MOUSEBUTTONDOWN:
				if not UI.buttons.get_selected() == None:
					if event.button == 1:
						UI.buttons.get_selected().run(event)
				else:
					UI.buttons.set_typing(False)
					UI.buttons.set_input_button(None)
			elif event.type == pg.VIDEORESIZE and settings.resizeable:
				render_blocks.mode_3D()
				Display.init((event.w, event.h))
				Textures.update_pixel_size()
				render_blocks.mode_2D()
			elif event.type == pg.KEYDOWN and not UI.buttons.is_typing():
				if event.key == pg.K_F2:
					screenshot()

	glBegin(GL_QUADS)

	for i in range(4):
		glTexCoord2fv(Textures.title_coords[i])
		glVertex2fv(character_coords[i] * 2 - 1)

	glEnd()

	glBindTexture(GL_TEXTURE_2D, Textures.text[0])

	UI.write("Loading...", (-0.1625, 0), 0.1, color=(1, 1, 1), shadow=True)
	UI.buttons = loading_buttons
	UI.render_buttons()

	pg.display.flip()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	glBindTexture(GL_TEXTURE_2D, Textures.terrain[0])
	glDisable(GL_BLEND)
	render_blocks.mode_3D()
	glEnableClientState(GL_NORMAL_ARRAY)
	pg.mouse.set_visible(False)
	pg.mouse.set_pos(Display.centre)



def justify_text(tup):
	x, y = tup
	y = round(y * Display.centre[1]) / Display.centre[1]
	x = round(x * Display.centre[0]) / Display.centre[0]
	return (x, y)
