# Compiler settings
CXX      := mpicxx
CXXFLAGS := -O3 -Wall -Wextra -std=c++11

# PETSc paths (inside the 64 bit Docker container)
PETSC_DIR  ?= /build/petsc
PETSC_ARCH ?= arch-linux-c-opt

# Include directories
INCLUDES := -I. -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include

# Link libraries
LIBS := -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -ltetgen -lmpi -lstdc++ -lm -ldl

# Add rpath so the executable finds shared libraries at runtime
LIBS += -Wl,-rpath=$(PETSC_DIR)/$(PETSC_ARCH)/lib

# Target
TARGET   := test_runner
SRCS     := BoxMeshDM_3D.cpp test_BoxMeshDM_3D.cpp
OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean test
