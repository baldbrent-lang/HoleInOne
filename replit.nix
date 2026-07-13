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
    # mediapipe pulls in opencv-contrib-python (the GUI build, unlike our
    # own opencv-python-headless), whose cv2 import dlopens libGL.so.1 and
    # libgthread-2.0.so.0. Without these the import dies and the pose
    # detector reports "mediapipe unavailable" even though the pip package
    # is installed.
    pkgs.libGL
    pkgs.glib
  ];
}
