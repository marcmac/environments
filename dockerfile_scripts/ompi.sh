#!/bin/bash

set -x

# Install the Mellanox OFED stack.  Note that this is dependent on what
# the base OS is (ie, Ubuntu 20.04) so if that changes then this needs updated.
#MOFED_VER=5.0-2.1.8.0
#MOFED_VER=5.5-1.0.3.2
MOFED_VER=5.4-3.4.0.0
OS_VER=$1
#OS_VER=ubuntu20.04
PLATFORM=x86_64
MOFED_TAR_URL="http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}"
MOFED_TAR="MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz"
TMP_INSTALL_DIR=/tmp/ofed

mkdir -p ${TMP_INSTALL_DIR}                                          && \
   cd ${TMP_INSTALL_DIR}                                             && \
   wget --quiet "${MOFED_TAR_URL}/${MOFED_TAR}"                      && \
   tar -xvf ${MOFED_TAR}                                             && \
   MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall   \
     --user-space-only --without-fw-update --all --force                \
     --skip-unsupported-devices-check                                && \
   rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz        \
          MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}            \
          MLNX_OFED_LINUX.*.logs                                     && \
   rm -rf ${TMP_INSTALL_DIR}
OFI=$2
if [ "$OFI" = "1" ]; then
  # Install OFI
  OFI_VER=1.15.1
  OFI_CONFIG_OPTIONS="--prefix ${OFI_INSTALL_DIR}"
  OFI_SRC_DIR=/tmp/ofi-src
  OFI_BASE_URL="https://github.com/ofiwg/libfabric/releases/download"
  OFI_URL="${OFI_BASE_URL}/v${OFI_VER}/libfabric-${OFI_VER}.tar.bz2"

  mkdir -p ${OFI_SRC_DIR}                              && \
    cd ${OFI_SRC_DIR}                                  && \
    wget ${OFI_URL}                                    && \
    tar -xf libfabric-${OFI_VER}.tar.bz2              && \
    cd libfabric-${OFI_VER}                            && \
    ./configure ${OFI_CONFIG_OPTIONS}                  && \
    make install                                       && \
    cd /tmp                                            && \
    rm -rf ${OFI_SRC_DIR}

  #OMPI CONFIG ARGS FOR OFI
  OMPI_CONFIG_OPTIONS_VAR="--prefix ${OMPI_INSTALL_DIR} --enable-shared --with-verbs --with-cma --with-pic --enable-mpi-cxx --enable-mpi-thread-multiple --with-pmi --with-pmix=internal --with-platform=contrib/platform/mellanox/optimized --disable-ucx --with-libfabric=/container/ofi"

else
  # Install UCX
  UCX_VER=1.10.1
  UCX_CONFIG_OPTIONS="--prefix ${UCX_INSTALL_DIR} --enable-mt"
  UCX_SRC_DIR=/tmp/ucx-src
  UCX_BASE_URL="https://github.com/openucx/ucx/releases/download"
  UCX_URL="${UCX_BASE_URL}/v${UCX_VER}/ucx-${UCX_VER}.tar.gz"

  mkdir -p ${UCX_SRC_DIR}                              && \
    cd ${UCX_SRC_DIR}                                  && \
    wget ${UCX_URL}                                    && \
    tar -xzf ucx-${UCX_VER}.tar.gz                     && \
    cd ucx-${UCX_VER}                                  && \
    ./contrib/configure-release ${UCX_CONFIG_OPTIONS}  && \
    make -j8 install                                   && \
    cd /tmp                                            && \
    rm -rf ${UCX_SRC_DIR}

  #OMPI CONFIG ARGS FOR UCX
  OMPI_CONFIG_OPTIONS_VAR="--prefix ${OMPI_INSTALL_DIR} --enable-shared --with-verbs --with-cma --with-pic --enable-mpi-cxx --enable-mpi-thread-multiple --with-pmi --with-pmix=internal --with-platform=contrib/platform/mellanox/optimized --with-ucx=/container/ucx"

fi

# Install OMPI
OMPI_VER=v4.1
OMPI_VER_NUM=4.1.0
OMPI_CONFIG_OPTIONS=${OMPI_CONFIG_OPTIONS_VAR}
OMPI_SRC_DIR=/tmp/openmpi-src
OMPI_BASE_URL="https://download.open-mpi.org/release/open-mpi"
OMPI_URL="${OMPI_BASE_URL}/${OMPI_VER}/openmpi-${OMPI_VER_NUM}.tar.gz"

mkdir -p ${OMPI_SRC_DIR}                        && \
  cd ${OMPI_SRC_DIR}                            && \
  wget ${OMPI_URL}                              && \
  tar -xzf openmpi-${OMPI_VER_NUM}.tar.gz       && \
  cd openmpi-${OMPI_VER_NUM}                    && \
  ./configure ${OMPI_CONFIG_OPTIONS}            && \
  make                                          && \
  make install                                  && \
  rm -rf ${OMPI_SRC_DIR}