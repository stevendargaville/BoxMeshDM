# Compiler settings
CC  := mpicc
CXX := mpicxx

# Standardized paths across Docker and Apptainer containers
PETSC_DIR  := /build/petsc
PETSC_ARCH := arch-linux-c-opt
TETGEN_INC := $(PETSC_DIR)/$(PETSC_ARCH)/externalpackages/tetgen1.6.0

# Compiler and Linker flags
INCLUDES := -I. -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(TETGEN_INC)

LIBS := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -ltetgen -lmpi -lstdc++ -lstdc++fs -lm -ldl
LIBS += -Wl,-rpath=$(PETSC_DIR)/$(PETSC_ARCH)/lib

# Container detection for leak sanitizer
ifneq ($(APPTAINER_CONTAINER),)
    # On a HPC cluster: Turn off leak checking to avoid ptrace crash
    export ASAN_OPTIONS := detect_leaks=0
    export LSAN_OPTIONS := exitcode=0
else
    # On your laptop: Keep leak tracking fully functional
    export ASAN_OPTIONS := detect_leaks=1
endif

# Build modes for C and C++
DEBUG_FLAGS_CXX   := -O0 -g -Wall -Wextra -std=c++17 -fsanitize=address,undefined -fPIC
RELEASE_FLAGS_CXX := -O3 -Wall -Wextra -std=c++17

DEBUG_FLAGS_C     := -O0 -g -Wall -Wextra -std=c11 -fsanitize=address,undefined -fPIC
RELEASE_FLAGS_C   := -O3 -Wall -Wextra -std=c11

# Default mode (release)
MODE ?= release

ifeq ($(MODE),debug)
    CXXFLAGS := $(DEBUG_FLAGS_CXX)
    CFLAGS   := $(DEBUG_FLAGS_C)
    # Explicitly force-link sanitizer runtimes if the mpicxx wrapper drops them
    LIBS += -lasan -lubsan
else
    CXXFLAGS := $(RELEASE_FLAGS_CXX)
    CFLAGS   := $(RELEASE_FLAGS_C)
endif

# MPI Execution setup
MPIEXEC ?= mpiexec
# MPICH oversubscribes by default
# -bind-to none to prevent core affinity limits
MPIEXEC_FLAGS ?= -bind-to none

# Targets and rules
TARGET_CXX := test_runner
TARGET_C   := c_test_runner

# Source files
LIB_SRC    := BoxMeshDM_3D.cpp
LIB_OBJ    := $(LIB_SRC:.cpp=.o)

SRCS_CXX   := test_BoxMeshDM_3D.cpp
OBJS_CXX   := $(SRCS_CXX:.cpp=.o)

SRCS_C     := test_lib_3D.c
OBJS_C     := $(SRCS_C:.c=.o)

all: $(TARGET_CXX) $(TARGET_C)

# Builds C++ Test Suite
$(TARGET_CXX): $(LIB_OBJ) $(OBJS_CXX)
	$(CXX) $(CXXFLAGS) $(LIB_OBJ) $(OBJS_CXX) $(LIBS) -o $(TARGET_CXX)

# Build C Integration Test (Link with CXX to ensure C++ runtime/std linkage)
$(TARGET_C): $(LIB_OBJ) $(OBJS_C)
	$(CXX) $(CXXFLAGS) $(LIB_OBJ) $(OBJS_C) $(LIBS) -o $(TARGET_C)

# Object compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Run test file across varying MPI process sizes + the C compatibility test
test: test_1 test_2 test_4 test_8 test_c

test_1: $(TARGET_CXX)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 1 MPI RANK"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 1 ./$(TARGET_CXX)

test_2: $(TARGET_CXX)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 2 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 ./$(TARGET_CXX)

test_4: $(TARGET_CXX)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 4 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 ./$(TARGET_CXX)

test_8: $(TARGET_CXX)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 8 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 8 ./$(TARGET_CXX)

test_c: $(TARGET_C)
	@echo "\n========================================="
	@echo " RUNNING PURE C INTEGRATION TEST"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 1 ./$(TARGET_C)

# Build modes
debug:
	$(MAKE) clean
	$(MAKE) MODE=debug

release:
	$(MAKE) clean
	$(MAKE) MODE=release

clean:
	rm -f $(LIB_OBJ) $(OBJS_CXX) $(OBJS_C) $(TARGET_CXX) $(TARGET_C)
	rm -rf test_outputs/

.PHONY: all clean test test_1 test_2 test_4 test_8 test_c debug release
