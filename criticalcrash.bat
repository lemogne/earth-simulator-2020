@echo off
color 97
setlocal EnableDelayedExpansion
title Your game has crashed very badly
goto error%1
:error
echo This is the file responsible for handling the most serious crashes of the game.
echo It is not supposed to be run externally, but doing so does no harm.
pause>nul
exit /b
:error0
echo You have not installed the necessary modules for the game to run!
echo Make sure you have Pygame, PyOpenGL and Numpy installed!
echo.
echo pip install pygame pyopengl pyopengl_accelerate numpy
goto readerror
:error1
echo The game's font files are missing!
goto readerror
:error2
echo The game's critical internal files are missing!
:readerror
echo.
set message=%*
set message=%message:~2%
for %%G IN (%message%) DO (
	set temp=%%G
	echo.!temp:"=!
)
pause>nul
exit