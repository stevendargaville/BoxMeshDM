# ~~~~~~~~~~~~~~~~~
# BoxMeshDM - Steven Dargaville
# Makefile for BoxMeshDM
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.24.0
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.24)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. Requires at least version 3.24.0)
endif

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Check if petsc has been configured with various options
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Read petscconf.h via awk (portable on macOS)
define _have_conf
$(shell awk '/^[[:space:]]*#define[[:space:]]+$(1)[[:space:]]+1/{print 1; exit}' $(PETSCCONF_H))
endef

# Check for Triangle support
export PETSC_HAVE_TRIANGLE := $(if $(call _have_conf,PETSC_HAVE_TRIANGLE),1,0)
ifeq ($(PETSC_HAVE_TRIANGLE),0)
$(error PETSc has not been configured with Triangle support. Reconfigure PETSc with --download-triangle)
endif
export PETSC_USE_SHARED_LIBRARIES := $(if $(call _have_conf,PETSC_USE_SHARED_LIBRARIES),1,0)

# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

# On macOS, strip any -Wl,-rpath,* when linking the shared library to avoid duplicate LC_RPATH
ifeq ($(shell uname -s 2>/dev/null),Darwin)
PETSC_LINK_LIBS_NORPATH := $(strip $(foreach w,$(LDLIBS),$(if $(findstring -Wl,-rpath,$(w)),,$(w))))
else
PETSC_LINK_LIBS_NORPATH := $(LDLIBS)
endif

# Output executable name
OUT := BoxMeshDM

# Output the library - either static or dynamic
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
LIB_OUT = libboxmeshdm.a
else
# mac osx name is different
ifeq ($(shell uname -s 2>/dev/null),Darwin)
LIB_OUT = libboxmeshdm.dylib
else
LIB_OUT = libboxmeshdm.so
endif
endif

# All the files required by BoxMeshDM - the library is built from these, the
# executable also links in BoxMeshDM_main.o, which is the only object with main
OBJS := BoxMeshDM.o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
# This builds the executable with main in it
all: $(OUT)

$(OUT): $(OBJS) BoxMeshDM_main.o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the library
lib: $(LIB_OUT)

$(LIB_OUT): $(OBJS)
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)	
	$(AR) $(AR_FLAGS) $(LIB_OUT) $(OBJS)
	$(RANLIB) $(LIB_OUT)
else
ifeq ($(shell uname -s 2>/dev/null),Darwin)
# macOS: Use -dynamiclib and set a relocatable @rpath install_name. Do not embed rpaths.
	$(LINK.F) -dynamiclib -o $(LIB_OUT) $(OBJS) $(PETSC_LINK_LIBS_NORPATH) -install_name @rpath/$(notdir $(LIB_OUT))
else	
# Linux: Use -shared and set the soname.
	$(LINK.F) -shared -o $(LIB_OUT) $(OBJS) $(PETSC_LINK_LIBS) -Wl,-soname,$(notdir $(LIB_OUT))
endif
endif

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Test library functionality
test_lib: test_lib.o $(LIB_OUT)
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
# Static library
	$(LINK.cc) -o test_lib test_lib.o $(LIB_OUT) $(PETSC_LIB)
else
# Shared library
ifeq ($(shell uname -s 2>/dev/null),Darwin)
# macOS: Link with rpath to find the dylib in current directory
	$(LINK.cc) -o test_lib test_lib.o -L. -lboxmeshdm $(PETSC_LIB) -Wl,-rpath,@loader_path
else
# Linux: Link with rpath to find the .so in current directory
	$(LINK.cc) -o test_lib test_lib.o -L. -lboxmeshdm $(PETSC_LIB) -Wl,-rpath,'$$ORIGIN'
endif
endif

# Print per-rank hashes of a generated mesh, to compare two builds of the library bit for bit
mesh_checksum: mesh_checksum.o $(LIB_OUT)
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
	$(LINK.cc) -o mesh_checksum mesh_checksum.o $(LIB_OUT) $(PETSC_LIB)
else
ifeq ($(shell uname -s 2>/dev/null),Darwin)
	$(LINK.cc) -o mesh_checksum mesh_checksum.o -L. -lboxmeshdm $(PETSC_LIB) -Wl,-rpath,@loader_path
else
	$(LINK.cc) -o mesh_checksum mesh_checksum.o -L. -lboxmeshdm $(PETSC_LIB) -Wl,-rpath,'$$ORIGIN'
endif
endif

tests_lib: test_lib
	@echo "Running tests on library..."
	./test_lib
	$(MPIEXEC) -n 2 ./test_lib 

# Tests - check executable exists and run it
# and also builds the library and tests it can be linked
# against and run
tests: BoxMeshDM
	@echo "Running tests on executable..."
	./BoxMeshDM
	./BoxMeshDM -target_edge_length 0.002
	./BoxMeshDM -target_edge_length 0.003 -final_smooth_its 5
	./BoxMeshDM -target_edge_length 0.004 -integrity_check 0
	./BoxMeshDM -target_edge_length 0.005 -print_stats 0
	./BoxMeshDM -target_edge_length 0.006 -integrity_check 0 -print_stats 0
	./BoxMeshDM -target_edge_length 0.005 -domain_width 2.0 -domain_height 0.5
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 0.01 -domain_width 1.0 -domain_height 1.0
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 0.005 -domain_width 2.0 -domain_height 0.5
# Same number of points as a unit square with edge length 0.01 - integrity check tolerances must scale
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 1e5 -domain_width 1e7 -domain_height 1e7
# Tiny edge length on a tiny domain - only 250 cells per side so within the index limit
	./BoxMeshDM -target_edge_length 4e-10 -domain_width 1e-7 -domain_height 1e-7
# 1/0.0099 is not whole - stepping the wall points by the edge length left a sliver at the corner
	./BoxMeshDM -target_edge_length 0.0099
	./BoxMeshDM -target_edge_length 0.005 -agglomeration_factor 1
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 0.005 -agglomeration_factor 2
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 0.005 -domain_width 2.0 -domain_height 0.5 -agglomeration_factor 2
# Thin domain - the halo is wider than half the tile height, but y is not split between ranks
	$(MPIEXEC) -n 2 ./BoxMeshDM -target_edge_length 0.005 -domain_width 2.0 -domain_height 0.1
	$(MAKE) lib
	$(MAKE) tests_lib
	@echo "All tests completed successfully!"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Cleanup
clean::
	$(RM) $(OUT) $(LIB_OUT) $(OBJS) BoxMeshDM_main.o test_lib test_lib.o mesh_checksum mesh_checksum.o *.dat