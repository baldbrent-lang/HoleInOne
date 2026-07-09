{ pkgs }: {
  deps = [
    pkgs.xorg.libXrender
    pkgs.xorg.libXext
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
    pkgs.sqlite-interactive
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.nodejs_20
    pkgs.postgresql
    pkgs.zlib
    pkgs.libjpeg
    pkgs.ffmpeg-headless
  ];
}
