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

# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

# Library to output
OUT := box_2D_gen_unstruc_mesh

# All the files required by box_2D_gen_unstruc_mesh
OBJS := box_2D_gen_unstruc_mesh.o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
all: $(OUT)

# Cleanup
clean::
	$(RM) box_2D_gen_unstruc_mesh
	$(RM) *.dat