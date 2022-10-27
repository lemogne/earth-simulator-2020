import numpy as np

#WORLD GENERATION
chunk_size = 16
world_height = 256
world_size_F = 5

heightlim = (10, 100)
Tres = (32, 32)
HLres = (8, 8)
Vres = (8, 8)
Gres = (2, 2)

tree_density = 1 / 128

#GRAPHICS
render_distance = 10
ticks_per_second = 15
maxFPS = 60
nominal_res = (1205, 720)
fullscreen = False		
fovY = 60
frame_cap = False
texture_pack = "default"		
shader_pack = "default"
schematic_pack = "default"
shadow_brightness = 2

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
shown = True

#OTHER
day_length = 1200  				# Length of a day in seconds
starting_time = -150  			# Time at which the game starts (midday=0; 1 day=1024)
unicode = True
letter_offset = 0.65
hitbox_epsilon = 0.00001
text_align = True
chunk_distance = lambda x, y: x + y	# Chunk distance norm

screenshot_dir = "screencaps"