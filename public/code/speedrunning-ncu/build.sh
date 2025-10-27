#!/bin/bash

# Build script for CUDA vector addition

set -e

BUILD_TYPE=${1:-Release}

echo "Building CUDA Vector Addition (${BUILD_TYPE})..."

# Create build directory
mkdir -p build
cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

# Build
cmake --build . -j$(nproc)

echo ""
echo "✓ Build complete!"
echo "  Executable: build/vector_add"
echo ""
echo "To run:"
echo "  cd build && ./vector_add"
echo ""
echo "To profile with NCU:"
echo "  cd build && ncu ./vector_add"


