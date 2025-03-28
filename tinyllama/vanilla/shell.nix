{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python311Full
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pytorch
    pkgs.python311Packages.pandas
    pkgs.python311Packages.pip
    pkgs.python311Packages.transformers
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      virtualenv .venv
      source .venv/bin/activate
    else
      source .venv/bin/activate
    fi
    echo "Welcome to your Python development environment."
  '';
}

