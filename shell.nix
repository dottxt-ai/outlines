{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

(pkgs.buildFHSEnv {
  name = "dottxt-ai";
  targetPkgs = pkgs:
    with pkgs; [
      autoconf
      binutils
      cmake
      cudatoolkit
      curl
      freeglut
      gcc13
      git
      gitRepo
      gnumake
      gnupg
      gperf
      libGL
      libGLU
      linuxPackages.nvidia_x11
      m4
      ncurses5
      procps
      python311
      stdenv.cc
      unzip
      util-linux
      uv
      xorg.libX11
      xorg.libXext
      xorg.libXi
      xorg.libXmu
      xorg.libXrandr
      xorg.libXv
      zlib
    ];

  multiPkgs = pkgs: with pkgs; [ zlib ];

  runScript = "bash";

  profile = ''
    # CUDA paths
    export CUDA_HOME=${pkgs.cudatoolkit}
    export CUDA_PATH=${pkgs.cudatoolkit}

    # Ensure proper binary paths are included
    export PATH=${pkgs.gcc13}/bin:${pkgs.cudatoolkit}/bin:$PATH

    # Set library paths, including additional directories for CUPTI
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib64:${pkgs.cudatoolkit}/extras/CUPTI/lib64:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH

    # Add static library paths to EXTRA_LDFLAGS for the linker
    export EXTRA_LDFLAGS="-L${pkgs.cudatoolkit}/lib64 -L${pkgs.cudatoolkit}/extras/CUPTI/lib64 -L${pkgs.linuxPackages.nvidia_x11}/lib -L${pkgs.cudatoolkit}/libdevice $EXTRA_LDFLAGS"
    export EXTRA_CCFLAGS="-I${pkgs.cudatoolkit}/include $EXTRA_CCFLAGS"

    # Set CMake paths
    export CMAKE_PREFIX_PATH=${pkgs.cudatoolkit}:${pkgs.linuxPackages.nvidia_x11}:$CMAKE_PREFIX_PATH

    # C++ and CC flags
    export CXXFLAGS="--std=c++17 $EXTRA_CCFLAGS"
    export CC=${pkgs.gcc13}/bin/gcc
    export CXX=${pkgs.gcc13}/bin/g++

    # NVCC flags to use the right compiler
    export NVCC_FLAGS="-ccbin ${pkgs.gcc13}/bin/gcc"
  '';

  structuredAttrs__ = {
    stdenv = pkgs.stdenv.overrideCC pkgs.stdenv.cc pkgs.gcc13;
  };
}).env
