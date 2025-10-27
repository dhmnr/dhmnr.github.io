@echo off
REM Build script for CUDA vector addition (Windows)

setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo Building CUDA Vector Addition (%BUILD_TYPE%)...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..

REM Build
cmake --build . --config %BUILD_TYPE%

echo.
echo Build complete!
echo   Executable: build\%BUILD_TYPE%\vector_add.exe
echo.
echo To run:
echo   cd build\%BUILD_TYPE% ^&^& vector_add.exe
echo.
echo To profile with NCU:
echo   cd build\%BUILD_TYPE% ^&^& ncu vector_add.exe

endlocal

