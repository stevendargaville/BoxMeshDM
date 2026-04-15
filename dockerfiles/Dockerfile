ARG BASE_IMAGE=stevendargaville/petsc

FROM ${BASE_IMAGE}
# Checkout the main branch by default
ARG BRANCH=main
# Use the optimised petsc build by default
ARG ARCH=arch-linux-c-opt

LABEL maintainer="Steven Dargaville"
LABEL description="BoxMeshDM"

ENV CHECKOUT_BRANCH=$BRANCH
ENV PETSC_ARCH=$ARCH
# -on_error_abort ensures any test failures are caught and the build fails
# fp trap turns on floating point exception trapping in petsc
ENV PETSC_OPTIONS="-on_error_abort -fp_trap on"

WORKDIR /build

# Ensure pipelines fail if any stage fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Copy the source code and build it
COPY . /build/BoxMeshDM

RUN set -e; \
    cd BoxMeshDM && \
    make clean && make lib && \
    make clean && make && \
    make clean && make tests

WORKDIR /build/BoxMeshDM
