{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.python311
    pkgs.python311Packages.torch-bin
    pkgs.python311Packages.torchaudio-bin
    pkgs.python311Packages.torch-audiomentations
    pkgs.python311Packages.librosa
    pkgs.python311Packages.jiwer
    pkgs.python311Packages.datasets
    pkgs.python311Packages.transformers
    pkgs.python311Packages.evaluate
    pkgs.python311Packages.accelerate
    pkgs.python311Packages.pip
    pkgs.python311Packages.pandas
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.numpy
    pkgs.python311Packages.faiss-cpu
    pkgs.python311Packages.sentence_transformers
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      virtualenv .venv
      source .venv/bin/activate
    else
      source .venv/bin/activate
    fi
    echo "Welcome to your Python development nix-shell."
    export CUDA_PATH=${pkgs.cudatoolkit}
    echo $CUDA_PATH
  '';
}

