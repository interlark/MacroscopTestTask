@echo off
for /D %%G in ("MovingObject*") do ..\x64\Release\TestMacroscop.exe "%~dp0%%G" 0.1 0.1 0.9 0.9
pause
