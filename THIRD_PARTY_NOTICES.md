# Data and dependencies

Project code is MIT licensed. The demonstration catalog is an excerpt of
[Amazon ESCI](https://github.com/amazon-science/esci-data), revision
`7916cdf6ab75a462e77f20ab40428a10923998d5`, under Apache License 2.0.
It contains historical English product text, not current inventory or offers.
The original license and notice are in `licenses/ESCI-LICENSE` and
`licenses/ESCI-NOTICE` and travel with the demonstration bundle.
Text fields are shortened for this experiment; no new relevance labels are added.

The external evaluation uses [Wayfair WANDS](https://github.com/wayfair/WANDS),
revision `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`, under its MIT license
(`licenses/WANDS-LICENSE`). WANDS product text is not included in the demonstration
catalog. The adapter preserves annotation provenance and reports its declared
deduplication policy. Raw datasets are downloaded separately, not committed.

LightGBM, scikit-learn, NumPy, SciPy, Polars, PyArrow, FastAPI, Uvicorn and
Joblib retain their own licenses. See the locked dependency list in `uv.lock`
and the installed distributions for their notices.

The interface uses system fonts and project-authored SVG/CSS. No third-party
product photographs, reviews, prices, brand logos or external font assets are used.
