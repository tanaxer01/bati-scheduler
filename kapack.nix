# test.nix
let
  kapack = import
    ( fetchTarball "https://github.com/oar-team/nur-kapack/archive/master.tar.gz") {};
in
  kapack.pkgs.mkShell {
    name = "testing-env";
    buildInputs =  [ kapack.batsim-master ];
  }
