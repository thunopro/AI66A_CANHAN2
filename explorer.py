import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import subprocess
    import polars as pl

    return pl, subprocess


@app.cell
def _(pl):
    pl.scan_csv("")
    return


@app.cell
def _(subprocess):
    subprocess.run(["wget","https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip"])
    subprocess.run(["wget","https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-002.zip"])
    return


if __name__ == "__main__":
    app.run()
