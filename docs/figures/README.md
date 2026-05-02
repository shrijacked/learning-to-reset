# Figures

Regenerate these SVG assets from local eval summaries with:

```bash
make figures
```

`scripts/build_figures.py` prefers metrics under `tmp/paper-eval/` and falls back
to the documented local pilot results when those files are absent. The figures are
checked in so the README renders on a fresh clone.
