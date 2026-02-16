from textures import *
import opensimplex as noise
import menu, init


def in_hitbox(r_pos, hitbox):
	b_pos = r_pos // 1
	return (r_pos >= b_pos + hitbox[0]).all() and (r_pos <= b_pos + hitbox[1]).all()


bytesneeded = lambda x: np.uint8(math.log(x, 256) + 1 // 1)

models = [Cube, Liquid, Slab, Layer, Ice, Air, Shrub]
block_models = np.array([5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 6, 3, 2])
non_solid = {0, 8, 18, 19}
non_jumpable = {0, 18, 19}
non_highlightable = {0, 8}
translucent = [0, 19]	# Must be list because of np.isin
side_full_array = np.array([model.is_side_full for model in models])

constV = ((0, 0), (0, 0), (0, 0))


def collision_check(pos, ds, dt):
	def collide(i, pos, ds, hitbox):
		if i == 1:
			offset = 0 if ds[i] > 0 else (1 - (settings.player_height % 1))
		else:
			offset = settings.player_width
			
		offset += settings.hitbox_epsilon
		velocity_offset = min(ds[i] * dt, 0.9)
		
		if ds[i] < 0:
			pos[i] = math.floor(pos[i] - velocity_offset + offset) + hitbox[0, i] - offset
		elif ds[i] > 0:
			pos[i] = math.floor(pos[i] - velocity_offset - offset) + hitbox[1, i] + offset
			
		ds[i] = 0

	# Check for block collisions
	segments = math.ceil(np.linalg.norm(ds * dt))
	for _ in range(segments):
		# Basic check in each dimension separately
		for i in range(3):
			if (hitbox := player.check_in_block(i, dt / segments, ds, pos)) is not None:
				collide(i, pos, ds, hitbox)

		# Edge cases
		while (hitbox := player.check_in_block(-1, dt / segments, ds, pos)) is not None and ds.any():
			collide(abs(ds).argmax(), pos, ds, hitbox)
		
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
		self.old_pos = self.pos + (0, 0, 0)
		block_under = World.get_block(self.pos - (0, 0.01, 0))
		hitbox_block = models[block_models[block_under]].hitbox
		
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
		accel[1] = downward * settings.jump_height * ((block_under not in non_jumpable) and in_hitbox(self.pos - (0, 0.01, 0), hitbox_block) or self.flying or self.mv[1] == 0)
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


	def not_in_hitbox(self, block, hitbox_min = None, hitbox_max = None):
		"""Checks if block at given location is not in hitbox of player."""
		if hitbox_min is None:
			hitbox_min = self.pos - (settings.player_width, 0, settings.player_width)
		if hitbox_max is None:
			hitbox_max = self.pos + (settings.player_width, settings.player_height, settings.player_width)
		
		hitbox_block = models[block_models[World.get_block(block)]].hitbox
		block_min = np.floor(block) + hitbox_block[0]
		block_max = block_min + hitbox_block[1]

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
					block = World.get_block((x, y, z))
					if block not in non_solid \
						and not self.not_in_hitbox((x, y, z), hitbox_min, hitbox_max):
						return models[block_models[block]].hitbox
		return None


	def rotate(self, mouse_pos):
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
				menu.Load_World.load_region(reg_pos)
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
			if World.seethrough[self.chunks[ch][:, y, :]].any():
				return y / World.chunk_size
		return World.height


	def thorough_chmax(self, ch):
		for y in range(World.height - 1, -1, -1):
			if World.seethrough[self.chunks[ch][:, y, :]].any():
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
	game_blocks = None
	seethrough = None
	biome_tint = None
	
	tropical_temp = 0.67
	desert_hum = 0.33
	taiga_temp = 0.33
	snow_temp = 0.2
	deep_snow_temp = 0.1

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


	def init(game_blocks, seethrough, biome_tint):
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
		
		World.game_blocks = game_blocks
		World.seethrough = seethrough
		World.biome_tint = biome_tint


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
			len(vert_tex_list) * init.types[settings.gpu_data_type][3],
			(init.types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list),
			GL_STATIC_DRAW
		)

		if chunkdata[1] != None:
			vert_tex_list = chunkdata[1][0]
			counter_transp = chunkdata[1][1]
			vbo_transp = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, vbo_transp)
			glBufferData(
				GL_ARRAY_BUFFER,
				len(vert_tex_list) * init.types[settings.gpu_data_type][3],
				(init.types[settings.gpu_data_type][2] * len(vert_tex_list))(*vert_tex_list), 
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
		return np.clip(np.column_stack((hum, temp)), 0, 1) #* init.types[settings.gpu_data_type][4]


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
		blocks_model = block_models[blocks]
		seethrough_copy = np.concatenate(([False], World.seethrough[1:]), 0)
		
		for i in range(7):
			# Basically, for each of the 6 possible faces of all cubes, we filter out all those, whose neighbour is not air;
			# the remainder we turn into vertex and texture data
			if i < 6:
				neighbours[i] = np.vstack(neighbours[i]).astype(int)
				light_neighb = np.vstack(
					np.reshape(
						np.tile(neighbours_light[i], World.height), 
						(World.chunk_size, World.height, World.chunk_size)
					)
				)

				neighb_transp = World.seethrough[neighbours[i]] | ~side_full_array[block_models[neighbours[i]], i]

				blocks_transp = blocks[neighb_transp]
				coords_transp = World.coord_array[neighb_transp]
				light_transp = light_neighb[neighb_transp]
				biome_transp = chunk_biome[neighb_transp]
				model_transp = blocks_model[neighb_transp]
			else:
				blocks_transp = blocks
				coords_transp = World.coord_array
				light_transp = light_neighb
				biome_transp = chunk_biome
				model_transp = blocks_model

			for j, model in enumerate(models):
				if model.triangles[i] is None:
					continue
				
				model_mask = model_transp == j

				solid_mask = ~World.seethrough[blocks_transp] & model_mask
				blocks_show = blocks_transp[solid_mask]  # Solid blocks
				coords_show = coords_transp[solid_mask]
				light_show = light_transp[solid_mask]
				biome_show = biome_transp[solid_mask]

				if i < 6:
					transp_mask = seethrough_copy[blocks] & (blocks != neighbours[i]) & (blocks_model == j)
					
					coords_show_transp = World.coord_array[transp_mask]
					blocks_show_transp = blocks[transp_mask]
					light_show_transp = light_neighb[transp_mask]
					biome_show_transp = chunk_biome[transp_mask]

					transp_mask2 = (neighbours[i] == 8) & (blocks == 0)
					
					coords_show_transp2 = World.coord_array[transp_mask2]
					blocks_show_transp2 = neighbours[i][transp_mask2]
					light_show_transp2 = light_neighb[transp_mask2]
					biome_show_transp2 = chunk_biome[transp_mask2]
					
					coords_show_transp = np.concatenate((coords_show_transp, coords_show_transp2), 0)
					blocks_show_transp = np.concatenate((blocks_show_transp, blocks_show_transp2), 0)
					light_show_transp = np.concatenate((light_show_transp, light_show_transp2), 0)
					biome_show_transp = np.concatenate((biome_show_transp, biome_show_transp2), 0)
				else:
					transp_mask = seethrough_copy[blocks] & (blocks_model == j)
					
					coords_show_transp = World.coord_array[transp_mask]
					blocks_show_transp = blocks[transp_mask]
					light_show_transp = light_neighb[transp_mask]
					biome_show_transp = chunk_biome[transp_mask]

				has_biometint = np.repeat(World.biome_tint[blocks_show][:, np.newaxis], 2, 1)
				has_biometint_transp = np.repeat(World.biome_tint[blocks_show_transp][:, np.newaxis], 2, 1)

				if len(coords_show) > 0:
					n = len(model.vertices[model.triangles[i]])
					c_show_r = np.repeat(coords_show, n, 0)
					cube_verts = np.tile(model.vertices[model.triangles[i]], (len(coords_show), 1))

					verts.append(c_show_r + cube_verts - (128, 128, 128))
					tex_verts.append(np.vstack(Textures.game_blocks[blocks_show, 6 * i:6 * i + 6]))
					humtemp = World.get_hum_temp(biome_show, coords_show[:, 1])
					biome_verts.append(np.vstack(np.repeat(
						has_biometint * humtemp - 30000 * ~has_biometint * np.ones(humtemp.shape)
					, n, 0)))
					normals.append(
						np.tile(
							init.types[settings.gpu_data_type][4] * model.normals[i], 
							(n * len(coords_show), 1)
						) * np.tile(
							np.repeat(
								((light_show <= coords_show[:, 1]) + settings.shadow_brightness) /
								(settings.shadow_brightness + 1), n
							), 
							(3, 1)
						).T
					)

					counter += len(coords_show) * n
					
				
				if len(coords_show_transp) > 0:
					n = len(model.vertices[model.triangles[i]])
					c_show_r = np.repeat(coords_show_transp, n, 0)
					cube_verts = np.tile(model.vertices[model.triangles[i]], (len(coords_show_transp), 1))

					transp_verts.append(c_show_r + cube_verts - (128, 128, 128))
					transp_tex_verts.append(np.vstack(Textures.game_blocks[blocks_show_transp, 6 * i:6 * i + 6]))
					humtemp = World.get_hum_temp(biome_show_transp, coords_show_transp[:, 1])
					transp_biome_verts.append(np.vstack(np.repeat(
						has_biometint_transp * humtemp - 30000 * ~has_biometint_transp * np.ones(humtemp.shape),
					 n, 0)))
					transp_normals.append(
						np.tile(
							init.types[settings.gpu_data_type][4] * model.normals[i], 
							(n * len(coords_show_transp), 1)
						) * np.tile(
							np.repeat(
								((light_show_transp <= coords_show_transp[:, 1]) + settings.shadow_brightness) /
								(settings.shadow_brightness + 1), n
							), 
							(3, 1)
						).T
					)

					counter_transp += len(coords_show_transp) * n
					
		vert_tex_list = np.ravel(
			np.column_stack((
				np.vstack(verts), 
				np.vstack(tex_verts), 
				np.vstack(biome_verts),
				np.vstack(normals)
			))
		).astype(init.types[settings.gpu_data_type][0])

		if counter_transp != 0:
			vert_tex_transp = np.ravel(
				np.column_stack((
					np.vstack(transp_verts), 
					np.vstack(transp_tex_verts),
					np.vstack(transp_biome_verts),
					np.vstack(transp_normals)
				))
			).astype(init.types[settings.gpu_data_type][0])

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
		return World.get_hum_temp(
			World.biomemap[
				tuple(coords[[0, 2]] // World.chunk_size)
			][
				tuple(coords[[0, 2]] % World.chunk_size)
			], coords[1]
		)[0] / init.types[settings.gpu_data_type][4]


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
		
		if block >= len(World.game_blocks) or block < 0:
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

		current_light = math.floor(
			region.light[ch][
				math.floor(coords[0] % World.chunk_size)
			][
				math.floor(coords[2] % World.chunk_size)
			]
		)
		if block in translucent and World.get_block(coords) != 0 and math.floor(coords[1]) == current_light:
			h = math.floor(coords[1]) - 1
			while World.get_block((coords[0], h, coords[2])) in translucent:
				if h < 0:
					break
				h -= 1
			else:
				region.light[ch][math.floor(coords[0] % World.chunk_size)][math.floor(coords[2] % World.chunk_size)] = h
		elif block not in translucent and coords[1] > current_light:
			region.light[ch][
				math.floor(coords[0] % World.chunk_size)
			][
				math.floor(coords[2] % World.chunk_size)
			] = coords[1]
		
		region.chunks[ch][
			math.floor(coords[0] % World.chunk_size)
		][
			math.floor(coords[1])
		][
			math.floor(coords[2] % World.chunk_size)
		] = block

		World.update_chunk_min_max(coords, block)


	def update_chunk_min_max(coords, block):
		ch = (coords[0] // World.chunk_size, coords[2] // World.chunk_size)
		region, ch = World.get_region(ch)
		
		if not region or not region.gen_chunks[ch]:
			return
		
		# Update chunk min and max values
		chmin, chmax = region.chunk_min_max[ch]
		
		if World.seethrough[block]:
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


	def get_biome_ident(biome):
		if biome[1] < World.deep_snow_temp:
			return "Very Cold Taiga"
		elif biome[1] < World.snow_temp:
			return "Cold Taiga"
		elif biome[1] < World.taiga_temp:
			return "Taiga"
		elif biome[1] < World.tropical_temp:
			return "Temperate Forest"
		elif biome[0] < World.desert_hum:
			return "Desert"
		else:
			return "Jungle"


	def get_temp_celsius(temp):
		tropical_temp_celsius = 35
		return tropical_temp_celsius * (temp - World.snow_temp) / (World.tropical_temp - World.snow_temp)


	def get_temp_fahrenheit(temp):
		tropical_temp_celsius = 95 - 32
		return 32 + tropical_temp_celsius * (temp - World.snow_temp) / (World.tropical_temp - World.snow_temp)
	
	
	def update_time():
		World.game_time = settings.starting_time + ((time.time() - init.Time.start) / settings.day_length) * 1024
	
	
	def get_24h_time():
		hrs = (World.game_time / 1024) * 24 + 12
		day = int(hrs // 24)
		hr = int(hrs % 24)
		mn = int((hrs % 1) * 60)
		return day, hr, mn


def compute_lighting(blocks):
	not_found = np.full((blocks.shape[0], blocks.shape[2]), True)
	light = np.full((blocks.shape[0], blocks.shape[2]), blocks.shape[1])
	for y in range(blocks.shape[1] - 1, -1, -1):
		not_found &= np.isin(blocks[:, y, :], translucent)
		light[not_found] = y - 1
	return light


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


player = Player()
player.init()


