@echo off
REM NCU profiling script for CUDA vector addition (Windows)

setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

set EXECUTABLE=build\%BUILD_TYPE%\vector_add.exe

if not exist %EXECUTABLE% (
    echo Error: %EXECUTABLE% not found. Please run build.bat first.
    exit /b 1
)

echo === Running NCU Profiling ===
echo.

echo 1. Basic Profile
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum %EXECUTABLE%

echo.
echo 2. Memory Metrics
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,l1tex__t_bytes.sum %EXECUTABLE%

echo.
echo 3. Compute Metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active %EXECUTABLE%

echo.
echo 4. Occupancy Analysis
ncu --metrics launch__occupancy_limit_blocks,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem %EXECUTABLE%

echo.
echo Profiling complete!
echo.
echo For detailed analysis, run:
echo   ncu --set full -o vector_add_profile %EXECUTABLE%
echo   ncu-ui vector_add_profile.ncu-rep

endlocal


