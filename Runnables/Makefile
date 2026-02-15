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

# Debug and release base options (SIMD-specific flags are appended via SIMD_FLAGS)
DEBUG = -rdynamic -Werror -Wno-pedantic -Wall -Wextra -Wparentheses \
        -Wfatal-errors -D_GLIBCXX_DEBUG -g -fno-omit-frame-pointer

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

# Prefetch / SIMD-related defines passed to source
PREFETCH = -DENABLE_PREFETCH -DUSE_SIMD
ifneq ($(IS_ARM),)
	PREFETCH += -DUSE_NEON
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
all: TREXRelease NetworkRelease

# -----------------------------
# Network
# -----------------------------
NetworkDebug:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(DEBUG) -o Network Network.cpp

NetworkRelease:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) -o Network Network.cpp

# -----------------------------
# TREX
# -----------------------------
TREXDebug:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(DEBUG) \
	      -o TREX TREX.cpp -ltbb -latomic $(PREFETCH)

TREXRelease:
	$(CC) $(FLAGS) $(OPTIMIZATION) $(RELEASE) \
	      -o TREX TREX.cpp -ltbb -latomic $(PREFETCH) -g

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
	rm -f TREX TREX_pgo_gen Network
	rm -rf $(PGO_DIR)

