# tumor-finder-2000

## How to run

On macOS, from the repository root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

`python -m app` opens the tumor-search window and loads `runs/best.pt`.

Train that checkpoint:

```
python -m app.train
```

Highlight one slice from the command line:

```
python -m app.highlight path/to/slice.jpg
```
