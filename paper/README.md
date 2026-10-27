# Concordat paper

`concordat.tex` and `references.bib` are the source; `concordat.pdf` is the built paper.

Build with [Tectonic](https://tectonic-typesetting.github.io/):

    tectonic -X compile --keep-intermediates concordat.tex

For arXiv, upload `concordat.tex`, `references.bib` and the generated `concordat.bbl`
(`tar czf arxiv-submission.tar.gz concordat.tex references.bib concordat.bbl`).
