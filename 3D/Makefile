# Compiler settings
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

# Build modes
DEBUG_FLAGS   := -O0 -g -Wall -Wextra -std=c++17 -fsanitize=address,undefined -fPIC
RELEASE_FLAGS := -O3 -Wall -Wextra -std=c++17

# Default mode (release)
MODE ?= release

ifeq ($(MODE),debug)
    CXXFLAGS := $(DEBUG_FLAGS)
    # Explicitly force-link sanitizer runtimes if the mpicxx wrapper drops them
    LIBS += -lasan -lubsan
else
    CXXFLAGS := $(RELEASE_FLAGS)
endif

# MPI Execution setup
MPIEXEC ?= mpiexec
# MPICH oversubscribes by default
# -bind-to none to prevent core affinity limits
MPIEXEC_FLAGS ?= -bind-to none

# Targets and rules
TARGET := test_runner
SRCS   := BoxMeshDM_3D.cpp test_BoxMeshDM_3D.cpp
OBJS   := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run test file across varying MPI process sizes
test: test_1 test_2 test_4 test_8

test_1: $(TARGET)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 1 MPI RANK"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 1 ./$(TARGET)

test_2: $(TARGET)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 2 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 2 ./$(TARGET)

test_4: $(TARGET)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 4 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 4 ./$(TARGET)

test_8: $(TARGET)
	@echo "\n========================================="
	@echo " RUNNING TESTS ON 8 MPI RANKS"
	@echo "========================================="
	$(MPIEXEC) $(MPIEXEC_FLAGS) -n 8 ./$(TARGET)

# Build modes
debug:
	$(MAKE) clean
	$(MAKE) MODE=debug

release:
	$(MAKE) clean
	$(MAKE) MODE=release

clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf test_outputs/

.PHONY: all clean test test_1 test_2 test_4 test_8 debug release
