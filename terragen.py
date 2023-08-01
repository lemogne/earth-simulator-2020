from init import *
import opensimplex as noise

def generate_perlin_noise_2d(coords, shape, res, seed=None):
	if seed == None:
		seed = World.seed

	seed += coords[0] * 58274
	seed += coords[1] * (-437)

	def f(t):
		return 6 * t**5 - 15 * t**4 + 10 * t**3

	delta = (res[0] / shape[0], res[1] / shape[1])
	d = (shape[0] // res[0], shape[1] // res[1])
	#print(d)
	#print(shape)
	#print(res)
	grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
	# Gradients
	angles = 2 * np.pi * rand(seed, (res[0] + 1, res[1] + 1), coords)
	gradients = np.dstack((np.cos(angles), np.sin(angles)))
	g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
	g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
	g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
	g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
	# Ramps
	n00 = np.sum(grid * g00, 2)
	n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
	n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
	n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
	# Interpolation
	t = f(grid)
	n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
	n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
	return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

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
	gen_heightmaps(np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]) + coords)
	heightmap = World.heightmap[coords]

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
	World.chunk_min_max[coords] = (chmin, (np.max(heightmap) / settings.chunk_size) - chmin)
	World.chunks[coords] = (8 * water_map + 2 * grass_map + 9 * sand_map + 3 * dirt_map +
												stone_map).astype(np.uint8)
	World.light[coords] = ((heightmap > settings.water_level) * heightmap + (heightmap < settings.water_level) * settings.water_level)
	pg.event.get()	# To prevent operating system from marking process as frozen

# Generate random array of chunks
def gen_terrain():
	WorldSize = np.array((2**settings.world_size_F, 2**settings.world_size_F))
	World.height = settings.world_height
	World.chunk_size = settings.chunk_size
	World.chunk_min_max = dict()

	World.chunks = {}
	World.light = {}
	make_coord_array()
	# Generate Chunk block arrays based on height map
	for _i in range(WorldSize[0]):
		for _k in range(WorldSize[1]):
			# Actually generate chunks and calculate lighting
			ch = (_i - WorldSize[0] // 2, _k - WorldSize[1] // 2)
			gen_chunk(ch)
			pg.event.get()	# To prevent operating system from marking process as frozen


	# Setup
	player.pos = np.array((
	    -0.5,
	    100.0,
		-0.5)
	)
	return
	# Generate tree map

	# -Coordinates and types for trees
	x = np.hstack(
	    rand(World.seed, (round(settings.tree_density * WorldSize[1] * WorldSize[0] * (settings.chunk_size**2)), 1)) *
	    (WorldSize[0] * settings.chunk_size - 8)).astype(int)
	z = np.hstack(
	    rand(World.seed + 89128,
	         (round(settings.tree_density * WorldSize[1] * WorldSize[0] * (settings.chunk_size**2)), 1)) *
	    (WorldSize[1] * settings.chunk_size - 8)).astype(int)
	t_type = np.hstack(
	    rand(World.seed + 928734, (round(settings.tree_density * WorldSize[1] * WorldSize[0] *
	                                     (settings.chunk_size**2)), 1)) * len(schematic["tree"])).astype(int)

	# -Create list of trees
	trees = np.array((x, heightmap[x + 2, z + 3] + 1, z, t_type)).T
	if len(trees) > 0:
		trees = trees[terrainmap[x + 2, z + 3]]
		trees = trees[trees[:, 1] > 35]

	# Paste Trees into terrain:
	for tree in trees:
		pg.event.get()
		for x in range(5):
			for y in range(7):
				for z in range(6):
					#print(schematic["tree"][tree[3]])
					if schematic["tree"][tree[3]][x, y, z] != 0:
						World.put_block(
						    (tree[0] + x - (WorldSize[0] / 2) * settings.chunk_size, tree[1] + y, tree[2] + z -
						     (WorldSize[1] / 2) * settings.chunk_size), schematic["tree"][tree[3]][x, y, z])

