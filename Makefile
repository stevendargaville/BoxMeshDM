# ~~~~~~~~~~~~~~~~~
# box_2D_gen_unstruc_mesh - Steven Dargaville
# Makefile for box_2D_gen_unstruc_mesh
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
OUT := box_2D_gen_unstruc_mesh

# Output the library - either static or dynamic
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
LIB_OUT = libbox_2D_gen_unstruc_mesh.a
else
# mac osx name is different
ifeq ($(shell uname -s 2>/dev/null),Darwin)
LIB_OUT = libbox_2D_gen_unstruc_mesh.dylib
else
LIB_OUT = libbox_2D_gen_unstruc_mesh.so
endif
endif

# All the files required by box_2D_gen_unstruc_mesh
OBJS := box_2D_gen_unstruc_mesh.o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
# This builds the executable with main in it
all: $(OUT)
override CXXFLAGS += -DSTANDALONE_MESH_GEN

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the library (either static or dynamic depending on what petsc was configured with)
lib: $(OBJS)
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

# Cleanup
clean::
	$(RM) $(OUT) $(LIB_OUT) $(OBJS) *.dat