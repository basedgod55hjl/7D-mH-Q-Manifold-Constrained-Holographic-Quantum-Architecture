@echo off
echo Fast cleaning...

echo Removing projects\web_hologram\node_modules...
if exist "projects\web_hologram\node_modules" rmdir /s /q "projects\web_hologram\node_modules"

echo Removing projects\web_hologram\.next...
if exist "projects\web_hologram\.next" rmdir /s /q "projects\web_hologram\.next"

echo Removing target...
if exist "target" rmdir /s /q "target"

echo Removing logs...
del /q *.log
del /q check_chat.txt
del /q check_chat_v2.txt
del /q verification_*.txt
del /q *.tmp

echo Fast Cleanup Done.
