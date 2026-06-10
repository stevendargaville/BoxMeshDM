# Compiler settings
CXX := mpicxx

# PETSc paths (inside Docker)
PETSC_DIR  ?= /build/petsc
PETSC_ARCH ?= arch-linux-c-opt

# Include directories
INCLUDES := -I. -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include

# Link libraries
LIBS := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -ltetgen -lmpi -lstdc++ -lm -ldl
LIBS += -Wl,-rpath=$(PETSC_DIR)/$(PETSC_ARCH)/lib

# Sources
TARGET := test_runner
SRCS   := BoxMeshDM_3D.cpp test_BoxMeshDM_3D.cpp
OBJS   := $(SRCS:.cpp=.o)

# Build modes
DEBUG_FLAGS   := -O0 -g -Wall -Wextra -std=c++11 -fsanitize=address,undefined
RELEASE_FLAGS := -O3 -Wall -Wextra -std=c++11

# Default mode (release)
MODE ?= release

ifeq ($(MODE),debug)
    CXXFLAGS := $(DEBUG_FLAGS)
else
    CXXFLAGS := $(RELEASE_FLAGS)
endif

# Targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run
test: $(TARGET)
	./$(TARGET)

# Build modes
debug:
	$(MAKE) clean
	$(MAKE) MODE=debug

release:
	$(MAKE) clean
	$(MAKE) MODE=release

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean test
