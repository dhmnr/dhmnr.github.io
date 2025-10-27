#!/bin/bash

# NCU profiling script for CUDA vector addition

set -e

if [ ! -f "build/vector_add" ]; then
    echo "Error: build/vector_add not found. Please run build.sh first."
    exit 1
fi

cd build

echo "=== Running NCU Profiling ==="
echo ""

# Basic profile
echo "1. Basic Profile"
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum ./vector_add

echo ""
echo "2. Memory Metrics"
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,l1tex__t_bytes.sum ./vector_add

echo ""
echo "3. Compute Metrics"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ./vector_add

echo ""
echo "4. Occupancy Analysis"
ncu --metrics launch__occupancy_limit_blocks,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem ./vector_add

echo ""
echo "✓ Profiling complete!"
echo ""
echo "For detailed analysis, run:"
echo "  cd build && ncu --set full -o vector_add_profile ./vector_add"
echo "  ncu-ui vector_add_profile.ncu-rep"


