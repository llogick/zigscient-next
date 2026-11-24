#!/bin/sh

# Requires cmake ninja-build

set -x
set -e

TARGET="aarch64-linux-musl"
MCPU="baseline"
CACHE_BASENAME="zig+llvm+lld+clang-$TARGET-0.18.0-dev.92+1e1964473"
PREFIX="$HOME/deps/$CACHE_BASENAME"
ZIG="$PREFIX/bin/zig"
SPIRV_TEST_EXECUTOR="$HOME/deps/zig-spirv-test-executor-20ed6f7"
SPIRV_LOCK_DIR="${ZSF_LOCK_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lock}/zig-spirv"

export PATH="$HOME/local/bin:$PATH"

# Override the cache directories because they won't actually help other CI runs
# which will be testing alternate versions of zig, and ultimately would just
# fill up space on the hard drive for no reason.
export ZIG_GLOBAL_CACHE_DIR="$PWD/zig-global-cache"
export ZIG_LOCAL_CACHE_DIR="$PWD/zig-local-cache"

mkdir -p build-debug
cd build-debug

cmake .. \
  -DCMAKE_INSTALL_PREFIX="stage3-debug" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$ZIG;cc;-target;$TARGET;-mcpu=$MCPU" \
  -DCMAKE_CXX_COMPILER="$ZIG;c++;-target;$TARGET;-mcpu=$MCPU" \
  -DZIG_TARGET_TRIPLE="$TARGET" \
  -DZIG_TARGET_MCPU="$MCPU" \
  -DZIG_STATIC=ON \
  -DZIG_NO_LIB=ON \
  -GNinja

ninja install

# Must be done after zig cc is finished.
export ZIG_LIB_DIR="$PWD/../lib"

SPIRV_TEST_EXECUTOR_OUT="$PWD/spirv-test-executor"
stage3-debug/bin/zig build \
  --build-file "$SPIRV_TEST_EXECUTOR/build.zig" \
  --prefix "$SPIRV_TEST_EXECUTOR_OUT" \
  --system "$SPIRV_TEST_EXECUTOR/zig-pkg"

mkdir -p "$SPIRV_LOCK_DIR"

stage3-debug/bin/zig build install test docs \
  --maxrss "${ZSF_MAX_RSS:-0}" \
  --prefix stage4-debug \
  --search-prefix "$PREFIX" \
  --test-timeout 3m \
  -Dversion-string="$(stage3-debug/bin/zig version)" \
  -Dtarget=$TARGET \
  -Dcpu=$MCPU \
  -Duse-zig-libcxx \
  -Denable-llvm \
  -Dno-lib \
  -Dspirv-test-executor="$SPIRV_TEST_EXECUTOR_OUT/bin/zig-spirv-test-executor" \
  -Dspirv-test-lock-dir="$SPIRV_LOCK_DIR" \
  ${ZSF_GPU_DRIVERS:+$(printf " -Dspirv-test-driver=%s" $ZSF_GPU_DRIVERS)} \
  -Dskip-non-native

stage4-debug/bin/zig test ../test/behavior.zig
