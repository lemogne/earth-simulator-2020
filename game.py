from render_blocks import *
from threading import Thread
import menu, terragen

while 1:
	UI.in_menu = True
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

	while not UI.in_menu:
		mouse_pos = pg.mouse.get_pos()
		# Measure time passed since last frame
		now = time.time()

		if not settings.frame_cap or settings.max_FPS * (now - Time.last_frame) >= 1:
			temp_pos = player.pos + (0,0,0)
			if not UI.paused:
				delta_t = now - Time.last_tick
				if delta_t < 0:
					ds = settings.ticks_per_second * (player.old_pos - player.pos)
				else:
					ds = player.mv + (0,0,0)

				collision_check(temp_pos, ds, delta_t)
										
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			player.rotate(mouse_pos)
			render_sky(Time.start)
			glPushMatrix()
			glTranslatef(-temp_pos[0], -temp_pos[1] - player.height, -temp_pos[2])
			render(chat_string)
			glPopMatrix()
			pg.display.flip()
			Time.last_frame = now
			Time.frames += 1

		while Time.last_tick < now:
			player.do_tick(1 / settings.ticks_per_second)
			Time.last_tick += 1 / settings.ticks_per_second
			if UI.show_game_info:
				looked_at_coords = get_looked_at()[0]
				if looked_at_coords is not None:
					lookedAt_str = str(looked_at_coords // 1)[1:-1]
				else:
					lookedAt_str = "None"
				chat_string = f"""Position:\t{str(np.round(player.pos, 4))[1:-1]}
Rotation:\t{str(np.round(player.rot, 4))[1:-1]}
FPS:\t\t{Time.prev_frames}
Looking at:\t{lookedAt_str} (ID: {World.get_block(looked_at_coords)})
World Seed:\t{World.seed}
Game Time:\t{round(World.game_time)}"""

		# Resets frame count
		if int(now) != Time.last_second:
			Time.prev_frames = Time.frames
			Time.frames = 0
			Time.last_second = int(now)

		if UI.paused:
			UI.check_hover(mouse_pos)
		if UI.buttons.is_typing():
			if UI.buttons.get_input_button() == None:
				chat_string = UI.input_text(chat_string, start=2)
				if not UI.buttons.is_typing():
					try:
						exec(chat_string[2:])
					except Exception as e:
						exception = f"{type(e).__name__}: {e}"
						chat_string += "\n" + exception
						print(exception)
			else:
				UI.buttons.get_input_button().run()
				if not UI.buttons.is_typing():
					UI.buttons.set_input_button(None)
		else:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					quit()
				elif event.type == pg.MOUSEBUTTONDOWN:
					if UI.paused:
						if UI.buttons.get_selected():
							if event.button == 1:
								UI.buttons.get_selected().run()
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
							World.set_block(get_looked_at()[1], settings.current_block)
				elif event.type == pg.VIDEORESIZE and settings.resizeable:
					Display.init((event.w, event.h))
					Textures.update_pixel_size()
				elif event.type == pg.KEYDOWN and not UI.buttons.is_typing():
					if event.key == pg.K_F2:
						screenshot()
					elif event.key == pg.K_F1:
						settings.shown ^= True
					elif event.key == pg.K_ESCAPE:
						toggle_menu()
					elif event.key == pg.K_m:
						if UI.show_game_info:
							chat_string = ""
						UI.show_game_info ^= True
					elif event.key == pg.K_f:
						player.flying ^= True
					elif event.key == pg.K_r:
						UI.buttons.set_typing(True)
						UI.show_game_info = False
						chat_string = "> _"
					elif event.key == pg.K_c:
						chat_string = ""
					elif event.key == pg.K_UP:
						settings.current_block += 1
						settings.current_block %= len(game_blocks)
					elif event.key == pg.K_DOWN:
						settings.current_block -= 1
						settings.current_block %= len(game_blocks)
	UI.in_menu = True
	chunk_loop.join()
	if world_infinite:
		gen_chunk_loop.join()
