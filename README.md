# tumor-finder-2000

## Demo

<video src="demo.mov" controls></video>

[demo.mov](demo.mov)

## How to run

On macOS, from the repository root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

`python -m app` opens the tumor-search window. Choose a slice (JPEG, PNG, BMP, TIFF, or WebP), then press the search button.

The class comes from `runs/best_classifier.pt`. On a tumor slice the highlighted region comes from `runs/best_highlighter.pt`: the area is tinted and framed, and the window names the finding (glioma, meningioma, or pituitary tumor). A slice with no tumor stays unmarked, and the window says none was found. After a tumor is found, choose another slice to search again.

Highlight one slice from the command line. The class and the highlight come from the same files as the window:

```
python -m app.highlight path/to/slice.jpg
```

The picture is written to `runs/highlighted.png`. `--output` sets another path.
