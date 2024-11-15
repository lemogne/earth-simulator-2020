from init import *
import opensimplex as noise

def f(t):
	return 6 * t**5 - 15 * t**4 + 10 * t**3

def gen_trees(coords):
	total_trees = []
	for i in range(coords[0] - 1, coords[0] + 2):
		for j in range(coords[1] - 1, coords[1] + 2):
			if (i, j) in World.trees:
				total_trees.append(World.trees[(i, j)])
				continue

			# Calculate trees in chunk
			avg_n = settings.chunk_size * settings.chunk_size * settings.tree_density_mean
			n = noise.noise2(i / settings.T_res[0], j / settings.T_res[1]) * settings.tree_density_var * avg_n
			n += settings.chunk_size * settings.chunk_size * settings.tree_density_mean

			# Generate tree coordinates
			k = np.arange(int(n))
			x = (noise.noise2array(i + 64324.4 - 434.23 * k, np.array((j + 728491.2,))) + 1) / 2
			z = (noise.noise2array(np.array((i - 46324.4,)), j - 278491.2 + 1874.283 * k).T + 1) / 2
			tree_type = noise.noise2array(i - 2361.53 + 71834.23 * k, np.array((j - 94829.4,))) > 0
			x = (f(x) * settings.chunk_size).astype(np.int64)
			z = (f(z) * settings.chunk_size).astype(np.int64)
			y = World.heightmap[(i, j)][x, z].astype(np.int64)

			trees = np.array((x + i * settings.chunk_size, y, z + j * settings.chunk_size, tree_type)).T[:, 0]
			World.trees[(i, j)] = trees
			total_trees.append(trees)
	trees = np.concatenate(total_trees, axis=0)
	return trees


def gen_heightmaps(coord_list):
	for coords in coord_list:
		t_coords = tuple(coords)
		if t_coords in World.heightmap:
			continue

		x = np.arange(World.chunk_size) + coords[1] * World.chunk_size
		z = np.arange(World.chunk_size) + coords[0] * World.chunk_size

		heightmap = (noise.noise2array(x / World.T_res[0], z / World.T_res[1]) + 1) / 2
		levelmap = (noise.noise2array(x / World.HL_res[0], z / World.HL_res[1]) + 1) / 2
		variancemap = (noise.noise2array((x / World.V_res[0]) - 36742.47, (z / World.V_res[1]) + 478234.389) + 1) / 2
		generalmap = (noise.noise2array(x / World.G_res[0], z / World.G_res[1]) + 1) / 2
		m1map = (noise.noise2array(x * 0.25, z * 0.25) + 1) * 0.003
		m2map = (noise.noise2array(x, z) + 1) * 0.001

		heightmap *= variancemap**2
		heightmap **= 0.5
		heightmap += levelmap
		heightmap /= 2
		heightmap += 2 * generalmap
		heightmap /= 3
		heightmap += m1map + m2map
		heightmap /= 1.008
		heightmap = f(heightmap)	# Push values further towards extremes (else the height values are too close to the mean)
		heightmap = (heightmap * (World.heightlim[1] - World.heightlim[0]) + World.heightlim[0])
		World.heightmap[t_coords] = heightmap

def gen_chunk(coords):
	gen_heightmaps(np.mgrid[-1:2, -1:2].T.reshape((9, 2)) + coords)
	heightmap = World.heightmap[coords]
	region, ch = World.get_region(coords)
	if region.gen_chunks[ch]:
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
	rockC = ((60 / heightmap)**2) - 0.1

	# Map of where grass should be (x and z coordinates)
	terrainmap = (slope[0] <= rockC) & (slope[1] <= rockC) & (slope[2] <= rockC) & (slope[3] <= rockC)
	heights = World.coord_array3[:, :, :, 1]  # Array of y-values for each block in the chunk

	# Cut-down heightmap copied for each y value
	current_height = np.transpose(np.tile(heightmap, (World.height, 1, 1)), (1, 0, 2)).astype(np.uint16)

	# Terrain map (of where grass should be) cut down to current chunk and copied for each y value
	tm = np.transpose(
		np.tile(
			terrainmap, (World.height, 1, 1)), (1, 0, 2))

	# Create masks for each generated block type
	block_map = heights <= current_height
	water_map = (heights > current_height) & (heights <= World.water_level)
	surface_map = heights == current_height
	grass_map = surface_map & (heights > 34) & tm
	sand_map = surface_map & (heights < 35)
	dirt_map = block_map & (heights > (current_height - 3)) & ~surface_map & tm
	stone_map = block_map & ~grass_map & ~sand_map & ~dirt_map

	# Actually generate chunks and calculate lighting
	chmin = np.min(heightmap) / World.chunk_size
	region.chunk_min_max[ch] = (chmin, (np.max(heightmap) / World.chunk_size) - chmin)
	region.chunks[ch] = (8 * water_map + 2 * grass_map + 9 * sand_map + 3 * dirt_map +
												stone_map).astype(np.uint8)
	region.light[ch] = ((heightmap > World.water_level) * heightmap + (heightmap < World.water_level) * World.water_level)

	trees = gen_trees(coords)

	if len(trees) > 0:
		trees = trees[trees[:, 1] > 35]

	# Paste Trees into terrain:
	for tree in trees:
		dim = schematic["tree"][tree[3]][0].shape
		if World.get_block(tree[:3]) not in [2, 3, 0]:
			continue
		tree[[0, 2]] -= np.array(coords) * World.chunk_size
		tree[:3] -= (dim[0] // 2, 0, dim[2] // 2)
		min_x = max(-tree[0], 0)
		max_x = min(dim[0], World.chunk_size - tree[0])
		min_z = max(-tree[2], 0)
		max_z = min(dim[2], World.chunk_size - tree[2])

		if max_x < min_x or max_z < min_z:
			continue

		tree_part  = schematic["tree"][tree[3]][0][min_x : max_x, :, min_z : max_z]
		tree_light = schematic["tree"][tree[3]][1][min_x : max_x,    min_z : max_z]
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
		while time.time() - Time.last_frame >= 1 / settings.min_FPS:
			if UI.in_menu:
				return
			time.sleep(0.1)
		gen_chunk(ch)

def gen_chunk_thread():
	try:
		while not UI.in_menu:
			gen_chunks()
			time.sleep(1)
	except Exception as e:
		World.thread_exception = e
		print("".join(traceback.format_exc()))