# Nebula LaTeX report

`main.tex` is the report entry point. The source is split by chapter under
`sections/`, and the two supplied vector diagrams are under `figures/`.

Build locally with a TeX Live or MiKTeX installation:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively, upload the complete `report/` directory to Overleaf and select
pdfLaTeX. Compile from inside `report/` so the relative paths to repository
results resolve correctly.

Before final submission, replace or extend the title-page metadata as required
by the institution and regenerate all numerical claims from a clean, committed
repository snapshot.
