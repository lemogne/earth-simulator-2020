from init import *


# Generate random array of chunks
def gen_terrain():
	WorldSize = Vector(2**settings.world_size_F, 2**settings.world_size_F)

	heightmap = ((generate_perlin_noise_2d(WorldSize * settings.chunk_size, settings.Tres) + 1) / 2)
	levelmap = (generate_perlin_noise_2d(WorldSize * settings.chunk_size, settings.HLres) + 1) / 2
	variancemap = (generate_perlin_noise_2d(WorldSize * settings.chunk_size, settings.Vres, World.seed + 89128) + 1) / 2
	generalmap = (generate_perlin_noise_2d(WorldSize * settings.chunk_size, settings.Gres, World.seed) + 1) / 2
	m1map = (generate_perlin_noise_2d(WorldSize * settings.chunk_size, (128, 128), World.seed) + 1) * 0.003
	m2map = (generate_perlin_noise_2d(WorldSize * settings.chunk_size, (512, 512), World.seed + 89128) + 1) * 0.001

	heightmap *= variancemap**2
	heightmap **= 0.5
	heightmap += levelmap
	heightmap /= 2
	heightmap += 2 * generalmap
	heightmap /= 3
	heightmap += m1map + m2map
	heightmap -= min(heightmap.ravel())
	heightmap /= max(heightmap.ravel())
	heightmap = (heightmap * (settings.heightlim[1] - settings.heightlim[0]) + settings.heightlim[0])

	# List of neighbour blocks in each direction (except up/down); is used to determine slope at each location
	slope = np.array([
	    np.hstack((heightmap[:, -1:], heightmap[:, :-1])),
	    np.hstack((heightmap[:, 1:], heightmap[:, :1])),
	    np.vstack((heightmap[1:, :], heightmap[:1, :])),
	    np.vstack((heightmap[-1:, :], heightmap[:-1, :]))
	])

	slope -= heightmap

	# Rock constant for each block; determines, based on elevation, from which slope onwards stone is generated instead of grass
	rockC = ((60 / heightmap)**2) - 0.1

	heightmap = heightmap.astype(np.uint8)  # Compression
	# Map of where grass should be (x and z coordinates)
	terrainmap = (slope[0] <= rockC) & (slope[1] <= rockC) & (slope[2] <= rockC) & (slope[3] <= rockC)

	World.chunks = {}
	World.light = {}
	# Generate Chunk block arrays based on height map
	for _i in range(WorldSize[0]):
		for _k in range(WorldSize[1]):
			heights = coordArray3[:, :, :, 1]  # Array of y-values for each block in the chunk

			# Translate generated data into usable form

			# heightmap cut down to current chunk
			hm = heightmap[_i * settings.chunk_size:(_i + 1) * settings.chunk_size,
			               _k * settings.chunk_size:(_k + 1) * settings.chunk_size]

			# Cut-down heightmap copied for each y value
			currentHeight = np.transpose(np.tile(hm, (settings.world_height, 1, 1)), (1, 0, 2))

			# Terrain map (of where grass should be) cut down to current chunk and copied for each y value
			tm = np.transpose(
			    np.tile(
			        terrainmap[_i * settings.chunk_size:(_i + 1) * settings.chunk_size,
			                   _k * settings.chunk_size:(_k + 1) * settings.chunk_size], (settings.world_height, 1, 1)), (1, 0, 2))

			# Create masks for each generated block type
			BlockMap = heights <= currentHeight
			WaterMap = (heights > currentHeight) & (heights < 34)
			SurfaceMap = heights == currentHeight
			GrassMap = SurfaceMap & (heights > 34) & tm
			SandMap = SurfaceMap & (heights < 35)
			DirtMap = BlockMap & (heights > (currentHeight - 3)) & ~SurfaceMap & tm
			StoneMap = BlockMap & ~GrassMap & ~SandMap & ~DirtMap

			# Actually generate chunks and calculate lighting
			World.chunks[(_i - WorldSize[0] // 2,
			              _k - WorldSize[1] // 2)] = (8 * WaterMap + 2 * GrassMap + 9 * SandMap + 3 * DirtMap +
			                                          StoneMap).astype(np.uint8)
			World.light[(_i - WorldSize[0] // 2, _k - WorldSize[1] // 2)] = ((hm > 33) * hm + (hm < 34) * 33)
			pg.event.get()	# To prevent operating system from marking process as frozen

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

	# Setup
	player.pos = Vector(
	    -0.5,
	    float(
	        max(heightmap[int(WorldSize[0] * settings.chunk_size / 2)][int(WorldSize[1] * settings.chunk_size /
	                                                                       2)], 33) + 2), -0.5)
