# Makefile with ARM detection and SIMD/compiler mapping
# Keeps original x86 configuration as default. On ARM (aarch64 / arm64)
# switches to g++-15 and maps SIMD/compiler flags to ARM-friendly ones.

# Detect architecture
ARCH := $(shell uname -m)
UNAME_S := $(shell uname -s)

# Consider common ARM identifiers
IS_ARM := $(filter aarch64 arm64 armv7l armv8l,$(ARCH))

# Default (x86) compiler and SIMD flags
CC_DEFAULT = g++ -fopenmp
SIMD_X86 = -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store

# ARM compiler and SIMD flags (use g++-15 on ARM as requested)
# -march=armv8.2-a is a reasonable target for recent ARM CPUs; +simd requests the
# architecture SIMD extension where supported. mfpu=neon is widely-supported on ARM.
# If you're targeting older 32-bit ARM, you may need to adjust flags (this Makefile
# assumes 64-bit aarch64/arm64 by default).
CC_ARM = g++-15 -fopenmp
SIMD_ARM =

# Pick compiler and SIMD flags based on detected arch
ifeq ($(IS_ARM),)
	CC = $(CC_DEFAULT)
	SIMD_FLAGS = $(SIMD_X86)
else
	CC = $(CC_ARM)
	SIMD_FLAGS = $(SIMD_ARM)
endif

# Print selection when running make (helps debugging)
$(info Detected platform: $(ARCH) / $(UNAME_S))
$(info Using compiler: $(CC))
$(info Using SIMD flags: $(SIMD_FLAGS))

# On macOS with Homebrew, g++ does not automatically search Homebrew's
# include/lib paths. Add them so TBB and other deps are found.
# We also detect the running macOS version so the linker doesn't warn about
# a deployment-target mismatch with Homebrew-built dylibs.
HOMEBREW_PREFIX =
MACOS_FLAGS =
ifneq ($(IS_ARM),)
  ifeq ($(UNAME_S),Darwin)
    HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)
    MACOS_VERSION := $(shell sw_vers -productVersion 2>/dev/null)
    ifneq ($(MACOS_VERSION),)
      MACOS_FLAGS += -mmacosx-version-min=$(MACOS_VERSION)
    endif
  endif
endif

# Basic flags
FLAGS = -std=c++2b -pipe
PROJECT_ROOT := $(shell pwd)/..
FLAGS += -I$(PROJECT_ROOT)
ifneq ($(HOMEBREW_PREFIX),)
  FLAGS += -I$(HOMEBREW_PREFIX)/include -L$(HOMEBREW_PREFIX)/lib
endif
ifneq ($(MACOS_FLAGS),)
  FLAGS += $(MACOS_FLAGS)
endif

ifeq ($(IS_ARM),)
OPTIMIZATION = -march=native -O3
else
OPTIMIZATION = -mcpu=native -O3
endif
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
OPTIMIZATION += -flto=$(NPROC)

# Debug compiler override (sanitizers on macOS need clang's runtime)
CC_DEBUG = $(CC)
DEBUG_SANITIZERS = -fsanitize=address,undefined
LLVM_CLANG =
ifeq ($(UNAME_S),Darwin)
  ifneq ($(HOMEBREW_PREFIX),)
    LLVM_CLANG := $(HOMEBREW_PREFIX)/opt/llvm/bin/clang++
  endif
  ifneq ($(LLVM_CLANG),)
    ifneq ($(wildcard $(LLVM_CLANG)),)
      CC_DEBUG = $(LLVM_CLANG) -fopenmp
      FLAGS += -I$(HOMEBREW_PREFIX)/opt/llvm/include -L$(HOMEBREW_PREFIX)/opt/llvm/lib
      FLAGS += -I$(HOMEBREW_PREFIX)/opt/libomp/include -L$(HOMEBREW_PREFIX)/opt/libomp/lib
    else
      DEBUG_SANITIZERS =
    endif
  else
    DEBUG_SANITIZERS =
  endif
endif

# Debug and release base options (SIMD-specific flags are appended via SIMD_FLAGS)
DEBUG = -rdynamic -Werror -Wno-pedantic -Wall -Wextra -Wparentheses \
        -Wfatal-errors -D_GLIBCXX_DEBUG -g -fno-omit-frame-pointer $(DEBUG_SANITIZERS)

# LTO flag adjustment for clang-based debug builds
DEBUG_OPTIMIZATION = $(OPTIMIZATION)
ifneq (,$(findstring clang,$(CC_DEBUG)))
  DEBUG_OPTIMIZATION := $(filter-out -flto=%,$(OPTIMIZATION)) -flto
endif

ifeq ($(IS_ARM),)
RELEASE_BASE = -ffast-math -ftree-vectorize -Wfatal-errors -DNDEBUG \
               -fomit-frame-pointer -mtune=native -fno-stack-protector \
               -funroll-loops -Wno-alloc-size-larger-than
else
RELEASE_BASE = -ffast-math -ftree-vectorize -Wfatal-errors -DNDEBUG \
               -fomit-frame-pointer -fno-stack-protector \
               -funroll-loops -Wno-alloc-size-larger-than
endif

RELEASE = $(RELEASE_BASE) $(SIMD_FLAGS)

# -----------------------------
# Profiling (Release with Frame Pointers & Debug Info)
# -----------------------------
ifeq ($(IS_ARM),)
PROFILE_BASE = -ffast-math -ftree-vectorize -Wfatal-errors -DNDEBUG \
               -fno-omit-frame-pointer -mtune=native -fno-stack-protector \
               -funroll-loops -Wno-alloc-size-larger-than
else
PROFILE_BASE = -ffast-math -ftree-vectorize -Wfatal-errors -DNDEBUG \
               -fno-omit-frame-pointer -fno-stack-protector \
               -funroll-loops -Wno-alloc-size-larger-than
endif

PROFILE = $(PROFILE_BASE) $(SIMD_FLAGS)

DynamicProfile:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(PROFILE) \
	      -o Dynamic Dynamic.cpp -ltbb -latomic $(PREFETCH) -g


# Prefetch / SIMD-related defines passed to source
PREFETCH = -DENABLE_PREFETCH -DUSE_SIMD
ifneq ($(IS_ARM),)
	PREFETCH += -DUSE_NEON
endif

# Optional: enable the compile-gated time-cutoff optimization in TransferUpdate
# Usage: make DynamicRelease DYN_TRANSFER_TIME_CUTOFF=1
ifdef DYN_TRANSFER_TIME_CUTOFF
	PREFETCH += -DDYN_TRANSFER_TIME_CUTOFF
endif

# Libraries: default assumes GCC; clang debug builds on macOS usually don't have libatomic
LIBS_DEFAULT = -ltbb -latomic
LIBS_DEBUG = $(LIBS_DEFAULT)
ifneq (,$(findstring clang,$(CC_DEBUG)))
	LIBS_DEBUG = -ltbb
endif

# SPARSEHASH = -I ../ExternalLibs/sparsehash/include

# -----------------------------
# PGO configuration
# -----------------------------
PGO_DIR = pgo-data
PGO_GEN = -fprofile-generate=$(PGO_DIR)
PGO_USE = -fprofile-use=$(PGO_DIR) -fprofile-correction

# -----------------------------
# Default targets
# -----------------------------
all: TREXRelease NetworkRelease DynamicRelease

# -----------------------------
# Network
# -----------------------------
NetworkDebug:
	$(CC_DEBUG) $(FLAGS) $(DEBUG_OPTIMIZATION) $(DEBUG) -o Network Network.cpp

NetworkRelease:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) -o Network Network.cpp

# -----------------------------
# TREX
# -----------------------------
TREXDebug:
	$(CC_DEBUG) $(FLAGS) $(DEBUG_OPTIMIZATION) $(DEBUG) \
	      -o TREX TREX.cpp $(LIBS_DEBUG) $(PREFETCH)

TREXRelease:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) \
	      -o TREX TREX.cpp -ltbb -latomic $(PREFETCH) -g

# -----------------------------
# Dynamic
# -----------------------------
DynamicDebug:
	$(CC_DEBUG) $(FLAGS) $(DEBUG_OPTIMIZATION) $(DEBUG) \
	      -o Dynamic Dynamic.cpp $(LIBS_DEBUG) $(PREFETCH)

DynamicRelease:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) \
	      -o Dynamic Dynamic.cpp -ltbb -latomic $(PREFETCH) -g

# -----------------------------
# TREX PGO
# -----------------------------

# Step 1: Instrumented build
TREXPGO_GEN:
	mkdir -p $(PGO_DIR)
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) $(PGO_GEN) \
	      -o TREX_pgo_gen TREX.cpp -ltbb -latomic $(PREFETCH)

# Step 2: Run workload manually
#   ./TREX_pgo_gen <args>

# Step 3: Final optimized build
TREXPGO_USE:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) $(PGO_USE) \
	      -o TREX TREX.cpp -ltbb -latomic $(PREFETCH)

# -----------------------------
# Cleanup
# -----------------------------
clean:
	rm -f TREX TREX_pgo_gen Network Dynamic
	rm -rf $(PGO_DIR)
