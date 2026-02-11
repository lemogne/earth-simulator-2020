from init import *

def f(t):
	return 6 * t**5 - 15 * t**4 + 10 * t**3

def get_tree_schem(biome):
	if biome[1] < 0.3:
		return "spruce"
	elif biome[1] < 0.67:
		return "tree"
	elif biome[0] < 0.33:
		return "desert"
	else:
		return "jungle"

def gen_trees(coords):
	total_trees = []
	for i in range(coords[0] - 1, coords[0] + 2):
		for j in range(coords[1] - 1, coords[1] + 2):
			if (i, j) in World.trees:
				total_trees.append(World.trees[(i, j)])
				continue

			# Calculate trees in chunk
			avg_n = settings.chunk_size * settings.chunk_size * settings.tree_density_mean
			n = noise.noise2(i / settings.tree_res[0], j / settings.tree_res[1]) * settings.tree_density_var * avg_n
			n += settings.chunk_size * settings.chunk_size * settings.tree_density_mean

			# Generate tree coordinates
			k = np.arange(int(n))
			x = (noise.noise2array(i + 64324.4 - 434.23 * k, np.array((j + 728491.2,))) + 1) / 2
			z = (noise.noise2array(np.array((i - 46324.4,)), j - 278491.2 + 1874.283 * k).T + 1) / 2
			tree_type = ((noise.noise2array(i - 2361.53 + 71834.23 * k, np.array((j - 94829.4,))) + 1) / 2 * 4294967295).astype(np.uint32)
			x = (f(x) * settings.chunk_size).astype(np.int64)
			z = (f(z) * settings.chunk_size).astype(np.int64)
			y = World.heightmap[(i, j)][x, z].astype(np.int64)

			trees = np.array((x + i * settings.chunk_size, y, z + j * settings.chunk_size, tree_type)).T[:, 0]
			World.trees[(i, j)] = trees
			total_trees.append(trees)
	trees = np.concatenate(total_trees, axis=0)
	return trees


def gen_heightmaps(coord_list):
	def combine_terrain(l_map, v_map):
		return (l_map * (v_map ** 2)) ** 0.5

	for coords in coord_list:
		t_coords = tuple(coords)
		if t_coords in World.heightmap:
			continue

		x = np.arange(World.chunk_size) + coords[1] * World.chunk_size
		z = np.arange(World.chunk_size) + coords[0] * World.chunk_size

		m0_map = (noise.noise2array(x * 0.125, z * 0.125) + 1) * 0.04
		t_map = (noise.noise2array(x / World.T_res[0], z / World.T_res[1]) + 1) / 2
		tv_map = (noise.noise2array((x / World.T_res[0]) - 36742.47, (z / World.T_res[1]) + 478234.389) + 1) / 2
		hl_map = (noise.noise2array(x / World.HL_res[0], z / World.HL_res[1]) + 1) / 2
		hlv_map = (noise.noise2array((x / World.HL_res[0]) - 36742.47, (z / World.HL_res[1]) + 478234.389) + 1) / 2
		g_map = (noise.noise2array(x / World.G_res[0], z / World.G_res[1]) + 1) / 2
		gv_map = (noise.noise2array((x / World.G_res[0]) - 36742.47, (z / World.G_res[1]) + 478234.389) + 1) / 2
		c_map = (noise.noise2array(x / World.C_res[0], z / World.C_res[1]) + 1) / 2
		cv_map = (noise.noise2array((x / World.C_res[0]) - 36742.47, (z / World.C_res[1]) + 478234.389) + 1) / 2
		m1_map = (noise.noise2array(x * 0.25, z * 0.25) + 1) * 0.004
		m2_map = (noise.noise2array(x, z) + 1) * 0.002

		heightmap = \
			  0.24 * f(combine_terrain(t_map, hlv_map)) \
			+ 0.29 * f(combine_terrain(hl_map, gv_map)) \
			+ 0.24 * f(combine_terrain(g_map, cv_map)) \
			+ 0.19 * f(c_map)
		heightmap += 0.08 * f(combine_terrain(m0_map, tv_map))
		heightmap /= 1.08
		heightmap += m1_map + m2_map
		heightmap /= 1.006

		heightmap = f(heightmap)	# Push values further towards extremes (else the height values are too close to the mean)
		heightmap = (heightmap * (World.heightlim[1] - World.heightlim[0]) + World.heightlim[0])
		World.heightmap[t_coords] = heightmap

		World.gen_biomemap(t_coords, x, z)
		

def gen_chunk(coords):
	gen_heightmaps(np.mgrid[-1:2, -1:2].T.reshape((9, 2)) + coords)
	heightmap = World.heightmap[coords]
	biomemap = World.biomemap[coords]
	hummap = World.get_hum_temp(biomemap, heightmap).reshape((World.chunk_size, 2, World.chunk_size)) / types[settings.gpu_data_type][4]
	region, ch = World.get_region(coords)
	if not region or region.gen_chunks[ch]:
		return

	# Neighbouring chunks
	left  = World.heightmap[(coords[0]-1, coords[1])]
	right = World.heightmap[(coords[0]+1, coords[1])]
	up    = World.heightmap[(coords[0], coords[1]+1)]
	down  = World.heightmap[(coords[0], coords[1]-1)]

	# List of neighbour blocks in each direction (except up/down); is used to determine slope at each location
	slope = np.array([
	    np.hstack((down[:, -1:], heightmap[:, :-1])),
	    np.hstack((heightmap[:, 1:], up[:, :1])),
	    np.vstack((heightmap[1:, :], right[:1, :])),
	    np.vstack((left[-1:, :], heightmap[:-1, :]))
	])

	slope -= heightmap

	# Rock constant for each block; determines, based on elevation, from which slope onwards stone is generated instead of grass
	rockC = (((45 + 40 * hummap[:, 1, :]) / heightmap)**2) - 0.1

	# Map of where grass should be (x and z coordinates)
	terrainmap = (slope[0] <= rockC) & (slope[1] <= rockC) & (slope[2] <= rockC) & (slope[3] <= rockC)
	heights = World.y_array  # Array of y-values for each block in the chunk
	heightmap_int = heightmap.astype(np.uint16)

	# Create masks for each generated block type
	if coords in World.blockmap:
		block_map = World.blockmap.pop(coords)
	else:
		block_map = heights <= heightmap
	water_map = (heights > heightmap) & (heights <= World.water_level)
	surface_map = heights == heightmap_int
	grass_map = surface_map & (heights > World.water_level + 1) & terrainmap
	desert_map = (hummap[:, 0, :] < 0.33) & (hummap[:, 1, :] > 0.67)
	terrain_depth = 6 * hummap[:, 1, :]
	cold_map = (hummap[:, 1, :] < 0.2)
	snow_map = grass_map & cold_map
	grass_map &= ~desert_map & ~cold_map
	ice_map = water_map & (heights == World.water_level) & cold_map
	water_map &= ~ice_map
	sand_map = surface_map & ((heights < World.water_level + 2) | desert_map)
	dirt_map = block_map & (heights > (heightmap - terrain_depth)) & ~surface_map & terrainmap
	sand_map |= dirt_map & desert_map
	dirt_map &= ~desert_map
	stone_map = block_map & ~grass_map & ~sand_map & ~dirt_map & ~snow_map

	# Actually generate chunks and calculate lighting
	chmin = np.min(heightmap_int) / World.chunk_size
	region.chunk_min_max[ch] = (chmin, (np.max(heightmap_int) / World.chunk_size) - chmin)
	region.chunks[ch] = (
	     8 * water_map \
	  +  2 * grass_map \
	  +  9 * sand_map \
	  +  3 * dirt_map \
	  +  1 * stone_map \
	  + 13 * snow_map \
	  + 14 * ice_map
	).astype(np.uint8).transpose(1, 0, 2)
	region.light[ch] = ((heightmap_int > World.water_level) * heightmap_int + (heightmap_int < World.water_level) * World.water_level)

	trees = gen_trees(coords)

	if len(trees) > 0:
		trees = trees[trees[:, 1] > 35]

	# Paste Trees into terrain:
	for tree in trees:
		biome = World.get_biome(tree[:3])
		schemtype = schematic[get_tree_schem(biome)]
		schem = schemtype[int(tree[3] / 4294967295 * len(schemtype))]
		dim = schem[0].shape
		block_under_tree = World.get_block(tree[:3])
		if block_under_tree in [3, 13, 9]:
			height = World.get_height(tree[[0, 2]])
			if height < World.water_level + 2:
				continue
			rock_const = ((60 / height)**2) - 0.1
			neighbours = np.array((World.get_height(tree[[0, 2]] - (1, 0)), World.get_height(tree[[0, 2]] + (1, 0)),
			              World.get_height(tree[[0, 2]] - (0, 1)), World.get_height(tree[[0, 2]] + (0, 1))))
			if ((neighbours - height) > rock_const).any():
				continue
		elif block_under_tree != 2:
			continue
		tree[[0, 2]] -= np.array(coords) * World.chunk_size
		tree[:3] -= schem[2]
		min_x = max(-tree[0], 0)
		max_x = min(dim[0], World.chunk_size - tree[0])
		min_z = max(-tree[2], 0)
		max_z = min(dim[2], World.chunk_size - tree[2])

		if max_x < min_x or max_z < min_z:
			continue

		tree_part  = schem[0][min_x : max_x, :, min_z : max_z]
		tree_light = schem[1][min_x : max_x,    min_z : max_z]
		min_x = max(tree[0], 0)
		max_x = min(tree[0] + dim[0], World.chunk_size)
		min_z = max(tree[2], 0)
		max_z = min(tree[2] + dim[2], World.chunk_size)

		not_air = tree_part != 0
		light_change = tree_light != -1

		region.chunks[ch][min_x : max_x, tree[1] + 1 : tree[1] + 1 + dim[1], min_z : max_z][not_air] = tree_part[not_air]
		region.light[ch][min_x : max_x, min_z : max_z][light_change] = tree_light[light_change] + tree[1] + 1
	region.gen_chunks[ch] = True


# Generate random array of chunks
def gen_terrain():
	World.height = settings.world_height
	World.chunk_size = settings.chunk_size
	World.heightmap = {}
	World.trees = {}
	noise.seed(int(World.seed))
	make_coord_array()
	gen_region((0, 0))

	# Setup
	chunk_x = settings.region_size // 2
	x = settings.chunk_size * chunk_x
	player.pos = np.array((x - 0.5, World.heightmap[(chunk_x, chunk_x)][0, 0] + 2, x - 0.5))
	player.chunkpos = player.pos // settings.chunk_size

def gen_region(reg):
	World.regions[reg] = Region(reg)
	x = reg[0] * World.region_size
	y = reg[1] * World.region_size
	for i in range(x, x + World.region_size):
		for j in range(y, y + World.region_size):
			gen_chunk((i, j))
			pg.event.get()

def gen_chunks():
	while (ch := World.chunks_to_generate.pop(0) if len(World.chunks_to_generate) > 0 else None):
		while settings.min_FPS and time.time() - Time.last_frame >= 1 / settings.min_FPS:
			if UI.in_menu:
				return
			time.sleep(0.1)
		if World.regions_to_load:
			Load_World.load_region(World.regions_to_load[0])
			World.regions_to_load.pop(0)
		else:
			gen_chunk(ch)

def gen_chunk_thread():
	try:
		while not UI.in_menu:
			gen_chunks()
			time.sleep(1)
	except Exception as e:
		World.thread_exception = e
		print("".join(traceback.format_exc()))