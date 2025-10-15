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
            python3
            (python3.withPackages (
              ps: with ps; [
                jupyterlab
                numpy
                pandas
                matplotlib
                jupytext
                langchain
                langchain-community
                langchain-google-genai
                streamlit
                pypdf
                sentence-transformers
              ]
            ))
          ];
        };
      }
    );
}
