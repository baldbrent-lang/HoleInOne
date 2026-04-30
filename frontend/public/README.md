# Background image

Drop a JPEG named `bg.jpg` here and it becomes the site-wide background.

Recommended dimensions: 1920x1080 or larger, landscape orientation.

The file is gitignored locally (so you don't have to commit a giant binary)
but Replit's working tree will pick it up — drop it in via the Replit file
tree on the left, or upload via Shell:

    curl -L "https://your-image-source/photo.jpg" -o frontend/public/bg.jpg

If the file is missing the page falls back to the emerald gradient.
