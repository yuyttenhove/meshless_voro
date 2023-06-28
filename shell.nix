{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/4fec046839e4c2f58a930d358d0e9f893ece59f5.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.cargo
    pkgs.rustc
  ];
  HDF5_DIR = pkgs.symlinkJoin { name = "hdf5"; paths = [ pkgs.hdf5_1_10 pkgs.hdf5_1_10.dev ]; };
}
