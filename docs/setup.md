# Setup help

The [README](../README.md) is the normal starting point. Run commands in the
repository folder, where `requirements.txt` and `run_rating_curve.py` are saved.

## Python is not found

Install Python 3.10 or later. On Windows, if `py` works but `python` does not,
use `py` in both commands:

```sh
py -m pip install -r requirements.txt
py run_rating_curve.py
```

## A library is not found in Spyder or VS Code

The editor may use a different Python environment from your terminal. In the
editor's Python console, run:

```python
import sys
print(sys.executable)
```

Install the requirements using that interpreter. For example, in PowerShell:

```powershell
& "C:/path/to/python.exe" -m pip install -r requirements.txt
```

Then restart the editor's Python console and run the script again.

## Optional: keep the libraries in a separate environment

From the repository folder:

```sh
python -m venv .venv
```

On Windows you can use the environment directly without activating it:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_rating_curve.py
```

On macOS/Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_rating_curve.py
```

Select this environment in your editor if you want to run there too.

## The data file or columns cannot be found

Edit `MEASUREMENTS` at the top of `run_rating_curve.py`. Use a path relative to
the script, or a full path with forward slashes. CSV headers should be
`date,wl,discharge`; daily files use `date,wl`. Export an Excel worksheet as CSV
first. Check the actual filename: Windows may hide the `.csv` extension.

## Too few measurements remain

Check the numbers and `DATE_FORMAT`. Missing or invalid values, invalid supplied
dates and negative discharge are excluded. At least 20 usable measurements are
required; measurements sharing a date stay together during validation, so
concentrating many rows on a few dates can leave too few training observations.
Use the [method guide](validated.md#input-requirements) to diagnose these cases.
