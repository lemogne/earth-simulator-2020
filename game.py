from render_blocks import *
from threading import Thread
import terragen


menu.UI.init_font()
menu.Settings.generate_ui()

while 1:
	menu.UI.in_menu = True
	menu.run()
	if World.new:
		terragen.gen_terrain()

	World.load_chunks(True)
	process_chunks(True)

	chunk_loop = Thread(target = chunk_thread, daemon = True)
	chunk_loop.start()

	world_infinite = World.infinite

	if world_infinite:
		gen_chunk_loop = Thread(target = terragen.gen_chunk_thread, daemon = True)
		gen_chunk_loop.start()

	Time.start = time.time()
	Time.last_tick = Time.start
	Time.last_frame = Time.start

	# FPS counting
	Time.last_second = int(Time.start)
	Time.frames = 0
	Time.prev_frames = 0

	while not menu.UI.in_menu:
		mouse_pos = pg.mouse.get_pos()
		# Measure time passed since last frame
		now = time.time()

		if not settings.frame_cap or settings.max_FPS * (now - Time.last_frame) >= 1:
			temp_pos = player.pos + (0,0,0)
			
			if not menu.UI.paused:
				delta_t = now - Time.last_tick
				
				if delta_t < 0:
					ds = settings.ticks_per_second * (player.old_pos - player.pos)
				else:
					ds = player.mv + (0,0,0)

				collision_check(temp_pos, ds, delta_t)
										
			glClear(GL_DEPTH_BUFFER_BIT)
			
			if not (menu.UI.paused or menu.UI.buttons.is_typing()):
				player.rotate(mouse_pos)
			
			glPushMatrix()
			glRotatef(player.rot[1], 0, 1, 0)
			glRotatef(player.rot[0], -player.norm[2], 0, player.norm[0])
			render_sky()
			glTranslatef(-temp_pos[0], -temp_pos[1] - player.height, -temp_pos[2])
			render(chat_string)
			glPopMatrix()
			pg.display.flip()
			Time.last_frame = now
			Time.frames += 1

		while Time.last_tick < now:
			if not (menu.UI.paused or menu.UI.buttons.is_typing()):
				player.do_tick(1 / settings.ticks_per_second)
			Time.last_tick += 1 / settings.ticks_per_second
			
			if menu.UI.show_game_info:
				looked_at_coords = get_looked_at()[0]
				
				if looked_at_coords is not None:
					lookedAt_str = str(looked_at_coords // 1)[1:-1]
				else:
					lookedAt_str = "None"
				biome_info = World.get_biome(player.pos.astype(np.int64))
				time_info = World.get_24h_time()
				chat_string = f"Position:\t{str(np.round(player.pos, 4))[1:-1]}\n"\
					f"Rotation:\t{str(np.round(player.rot, 4))[1:-1]}\n"\
					f"FPS:\t\t{Time.prev_frames}\n"\
					f"Looking at:\t{lookedAt_str} (ID: {World.get_block(looked_at_coords)})\n"\
					f"World Seed:\t{World.seed}\n"\
					f"Biome:\t{World.get_biome_ident(biome_info)} "\
					f"(Temp. {round(World.get_temp_celsius(biome_info[1]), 1)}°C; Hum. {round(biome_info[0] * 100, 1)}%)/"\
					f"{np.round(biome_info, 4)}\n"\
					f"Game Time:\t Day {time_info[0]}, {time_info[1]:02}:{time_info[2]:02} ({round(World.game_time)})"

		# Resets frame count
		if int(now) != Time.last_second:
			Time.prev_frames = Time.frames
			Time.frames = 0
			Time.last_second = int(now)

		if menu.UI.paused:
			menu.UI.check_hover(mouse_pos)
		
		for event in pg.event.get():
			if menu.UI.buttons.is_typing():
				if menu.UI.buttons.get_input_button() == None:
					chat_string = menu.UI.input_text(chat_string, event, start=2)
					if not menu.UI.buttons.is_typing():
						try:
							exec(chat_string[2:])
						except Exception as e:
							exception = f"{type(e).__name__}: {e}"
							chat_string += "\n" + exception
							print(exception)
				else:
					menu.UI.buttons.get_input_button().run(event)
					
					if not menu.UI.buttons.is_typing():
						menu.UI.buttons.set_input_button(None)
			if event.type == pg.QUIT:
				pg.quit()
				quit()
			elif event.type == pg.MOUSEBUTTONDOWN:
				if menu.UI.paused:
					if menu.UI.buttons.get_selected():
						if event.button == 1:
							menu.UI.buttons.get_selected().run(event)
					else:
						menu.UI.buttons.set_typing(False)
						menu.UI.buttons.set_input_button(None)
				else:
					#Place/Destroy blocks
					if event.button == 1:
						looked_at = get_looked_at()[0]
						if looked_at is not None:
							if 8 in [
								World.get_block(looked_at + (0, 1, 0)),
								World.get_block(looked_at + (1, 0, 0)),
								World.get_block(looked_at + (-1, 0, 0)),
								World.get_block(looked_at + (0, 0, 1)),
								World.get_block(looked_at + (0, 0, -1))
							]:
								World.set_block(looked_at, 8)
							else:
								World.set_block(looked_at, 0)
					elif event.button == 2:
						if (looked_at := get_looked_at()[0]) is not None:
							settings.current_block = World.get_block(looked_at)
					elif event.button == 3:
						looked_at = get_looked_at()[1]
						if looked_at is not None and player.not_in_hitbox(looked_at):
							World.set_block(looked_at, settings.current_block)
			elif event.type == pg.VIDEORESIZE and settings.resizeable:
				player.rot = np.array((0.0, 0.0, 0.0))
				Display.init((event.w, event.h))
				Textures.update_pixel_size()
			elif event.type == pg.KEYDOWN and not menu.UI.buttons.is_typing():
				if event.key == pg.K_F2:
					screenshot()
				elif event.key == pg.K_F1:
					settings.shown ^= True
				elif event.key == pg.K_ESCAPE:
					menu.toggle_menu(event)
				elif event.key == pg.K_m:
					if menu.UI.show_game_info:
						chat_string = ""
					menu.UI.show_game_info ^= True
				elif event.key == pg.K_f:
					player.flying ^= True
				elif event.key == pg.K_r:
					menu.UI.buttons.set_typing(True)
					menu.UI.show_game_info = False
					chat_string = "> "
				elif event.key == pg.K_c:
					chat_string = ""
				elif event.key == pg.K_UP:
					settings.current_block += 1
					settings.current_block %= len(game_blocks)
				elif event.key == pg.K_DOWN:
					settings.current_block -= 1
					settings.current_block %= len(game_blocks)
	
	menu.UI.in_menu = True
	chunk_loop.join()
	
	if world_infinite:
		gen_chunk_loop.join()
