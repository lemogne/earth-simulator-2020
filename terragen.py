from init import *
import opensimplex as noise

def gen_trees(coords):
	trees = []
	for i in range(coords[0] - 1, coords[0] + 2):
		for j in range(coords[1] - 1, coords[1] + 2):
			# Calculate trees in chunk
			avg_n = settings.chunk_size * settings.chunk_size * settings.tree_density_mean
			n = noise.noise2(i / settings.T_res[0], j / settings.T_res[1]) * settings.tree_density_var * avg_n
			n += settings.chunk_size * settings.chunk_size * settings.tree_density_mean

			# Generate tree coordinates
			for k in range(int(n)):
				x = (noise.noise2(i + 64324.4 - 434.23 * k, j + 728491.2 - 8174.283 * k) + 1) / 2
				z = (noise.noise2(i - 46324.4 + 344.23 * k, j - 278491.2 + 1874.283 * k) + 1) / 2
				tree_type = noise.noise2(i - 2361.53 + 71834.23 * k, j - 94829.4 + 33428.3 * k) > 0
				x = int(x * settings.chunk_size)
				z = int(z * settings.chunk_size)
				y = int(World.heightmap[(i, j)][x, z])
				trees.append((x + i * settings.chunk_size, y, z + j * settings.chunk_size, tree_type))
	return np.array(trees)


def gen_heightmaps(coord_list):
	def f(t):
		return 6 * t**5 - 15 * t**4 + 10 * t**3
	noise.seed(int(World.seed))
	for coords in coord_list:
		t_coords = tuple(coords)
		if t_coords in World.heightmap:
			continue

		x = np.arange(settings.chunk_size) + coords[1] * settings.chunk_size
		z = np.arange(settings.chunk_size) + coords[0] * settings.chunk_size

		heightmap = (noise.noise2array(x / settings.T_res[0], z / settings.T_res[1]) + 1) / 2
		levelmap = (noise.noise2array(x / settings.HL_res[0], z / settings.HL_res[1]) + 1) / 2
		variancemap = (noise.noise2array((x / settings.V_res[0]) - 36742.47, (z / settings.V_res[1]) + 478234.389) + 1) / 2
		generalmap = (noise.noise2array(x / settings.G_res[0], z / settings.G_res[1]) + 1) / 2
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
		heightmap = (heightmap * (settings.heightlim[1] - settings.heightlim[0]) + settings.heightlim[0])
		World.heightmap[t_coords] = heightmap

def gen_chunk(coords):
	gen_heightmaps(np.mgrid[-1:2, -1:2].T.reshape((9, 2)) + coords)
	heightmap = World.heightmap[coords]
	region, ch = World.get_region(coords)
	#print(region, ch)

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

	heights = coordArray3[:, :, :, 1]  # Array of y-values for each block in the chunk

	# Cut-down heightmap copied for each y value
	current_height = np.transpose(np.tile(heightmap, (settings.world_height, 1, 1)), (1, 0, 2)).astype(np.uint16)

	# Terrain map (of where grass should be) cut down to current chunk and copied for each y value
	tm = np.transpose(
		np.tile(
			terrainmap, (settings.world_height, 1, 1)), (1, 0, 2))

	# Create masks for each generated block type
	block_map = heights <= current_height
	water_map = (heights > current_height) & (heights <= settings.water_level)
	surface_map = heights == current_height
	grass_map = surface_map & (heights > 34) & tm
	sand_map = surface_map & (heights < 35)
	dirt_map = block_map & (heights > (current_height - 3)) & ~surface_map & tm
	stone_map = block_map & ~grass_map & ~sand_map & ~dirt_map

	# Actually generate chunks and calculate lighting
	chmin = np.min(heightmap) / settings.chunk_size
	region.chunk_min_max[ch] = (chmin, (np.max(heightmap) / settings.chunk_size) - chmin)
	region.chunks[ch] = (8 * water_map + 2 * grass_map + 9 * sand_map + 3 * dirt_map +
												stone_map).astype(np.uint8)
	region.light[ch] = ((heightmap > settings.water_level) * heightmap + (heightmap < settings.water_level) * settings.water_level)
	pg.event.get()	# To prevent operating system from marking process as frozen

	trees = gen_trees(coords)

	if len(trees) > 0:
		trees = trees[trees[:, 1] > 35]

	# Paste Trees into terrain:
	for tree in trees:
		pg.event.get()
		if World.get_block(tree[:3] + (-2, 0, -3)) != 2:
			continue
		for x in range(5):
			for y in range(7):
				for z in range(6):
					a_x = x + tree[0] - 2
					a_z = z + tree[2] - 3
					x_in_bounds = coords[0] * settings.chunk_size <= a_x < (coords[0] + 1) * settings.chunk_size
					z_in_bounds = coords[1] * settings.chunk_size <= a_z < (coords[1] + 1) * settings.chunk_size
					if schematic["tree"][tree[3]][x, y, z] != 0 and x_in_bounds and z_in_bounds:
						World.put_block((a_x, y + tree[1] + 1, a_z), schematic["tree"][tree[3]][x, y, z])

# Generate random array of chunks
def gen_terrain():
	WorldSize = np.array((2**settings.world_size_F, 2**settings.world_size_F))
	World.height = settings.world_height
	World.chunk_size = settings.chunk_size

	World.regions = {(0, 0): Region((0, 0)), (-1, 0): Region((-1, 0)), (0, -1): Region((0, -1)), (-1, -1): Region((-1, -1))}
	make_coord_array()
	# Generate Chunk block arrays based on height map
	for _i in range(WorldSize[0]):
		for _k in range(WorldSize[1]):
			# Actually generate chunks and calculate lighting
			ch = (_i - WorldSize[0] // 2, _k - WorldSize[1] // 2)
			gen_chunk(ch)
			pg.event.get()	# To prevent operating system from marking process as frozen


	# Setup
	player.pos = np.array((-0.5, World.heightmap[(0, 0)][0, 0] + 1,-0.5)
	)
