{ pkgs }: {
  deps = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.nodejs_20
    pkgs.postgresql
    pkgs.zlib
    pkgs.libjpeg
    pkgs.ffmpeg-headless
  ];
}
