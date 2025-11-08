{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            # 使用したいパッケージをここに追加
            python3
            (python3.withPackages (
              ps: with ps; [
                keras
                jupyterlab
                numpy
                pandas
                matplotlib
                scikit-learn
                tensorflow
                torch
                ipykernel
              ]
            ))
          ];
          shellHook = ''
            PROJECT_NAME_RAW=$(basename "$PWD")
            PROJECT_NAME=$(echo "$PROJECT_NAME_RAW" | tr '[:upper:]' '[:lower:]')
            KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$PROJECT_NAME"

            if [ ! -d "$KERNEL_DIR" ]; then
              echo ">>> [flake.nix shellHook] Jupyter kernel '$PROJECT_NAME' not found. Registering..."
              python3 -m ipykernel install --user --name="$PROJECT_NAME"
              echo ">>> [flake.nix shellHook] Kernel registered."
            fi
          '';
        };
      }
    );
}
