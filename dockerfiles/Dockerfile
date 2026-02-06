ARG BASE_IMAGE=stevendargaville/petsc

FROM ${BASE_IMAGE}
# Checkout the main branch by default
ARG BRANCH=main
# Use the optimised petsc build by default
ARG ARCH=arch-linux-c-opt
# Don't use malloc dump by default
ARG MALLOC_DUMP=false

LABEL maintainer="Steven Dargaville"
LABEL description="box_2D_gen_unstruc_mesh"

ENV CHECKOUT_BRANCH=$BRANCH
ENV PETSC_ARCH=$ARCH
# -on_error_abort ensures any test failures are caught and the build fails
# fp trap turns on floating point exception trapping in petsc
ENV PETSC_OPTIONS="-on_error_abort -fp_trap on"

WORKDIR /build

# Ensure pipelines fail if any stage fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Clone box_2D_gen_unstruc_mesh and run all the tests
RUN set -e; \
    echo "Cloning branch: ${CHECKOUT_BRANCH}" && \
    git clone --depth=1 --branch ${CHECKOUT_BRANCH} https://github.com/stevendargaville/box_2D_gen_unstruc_mesh.git && \
    cd box_2D_gen_unstruc_mesh && \
    make clean && make lib && \
    make clean && make && \
    make clean && make tests

WORKDIR /build/box_2D_gen_unstruc_mesh
