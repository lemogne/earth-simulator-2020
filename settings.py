import numpy as np

#WORLD
chunk_size = 16
world_height = 256
region_size = 32
infinite = True

#WORLD GENERATION
heightlim = (20, 110)
T_res = (16, 16)
HL_res = (64, 64)
G_res = (256, 256)		
C_res = (4096, 4096)
B_res = (1024, 1024)

tree_density_mean = 1 / 64
tree_density_var = 1.1
tree_res = (16, 16)
water_level = 33

#GRAPHICS
render_distance = 30
ticks_per_second = 15
max_FPS = 60
min_FPS = 30
nominal_res = (1205, 720)
fullscreen = False		
fov_Y = 60
frame_cap = False
texture_pack = "default"
shader_pack = "default"
schematic_pack = "default"
shadow_brightness = 2
resizeable = True

#PLAYER
movement_speed = 5.0
flying_speed = 20.0
gravity = 12.0
jump_height = 6.0
player_height = 1.6
player_width = 0.3
flying = False  				# is fly mode enabled at start
current_block = 5  				# block in hand at start
hand_reach = 4
mouse_sensitivity = 0.2

#GUI
icon_size = 0.2  				# current block indicator
icon_offset = (0.8, -0.8)
pause_menu_color = (0.3, 0.3, 0.3)
button_scale = 0.6
logo_scale = 1
show_crosshair = True
shown = True					#hidden
logo_shown = True				#hidden

#OTHER
day_length = 1200  				# Length of a day in seconds
starting_time = -150  			# Time at which the game starts (midday=0; 1 day=1024)
unicode = True
hitbox_epsilon = 0.00001
text_align = True
chunk_distance = lambda x: abs(x[0] + x[1])	#hidden; Chunk distance norm
gpu_data_type = 1
worlds_per_page = 5				# Amount of worlds per page in the Load World menu
screenshot_dir = "screencaps"
