# Signal Noise Reduction Analyzer

College project for Problem 20 — Data Averaging and Interpolation.

## How to run

### Python scripts
install dependencies first:
pip install numpy scipy matplotlib plotly

then run:
python python/signal_gen.py
python python/filters.py
python python/matplotlib_viz.py
python python/plotly_viz.py

outputs get saved to analysis_output/ folder

### Web app
just open web/index.html in any browser. no setup needed.

## Project structure

signal-analyzer/
├── python/
│   ├── signal_gen.py       generates signals and noise
│   ├── filters.py          moving average, gaussian, median, savgol
│   ├── matplotlib_viz.py   static plots
│   └── plotly_viz.py       interactive html chart
├── web/
│   ├── index.html
│   ├── style.css
│   └── app.js              plotly chart + three.js 3d spectrum
├── analysis_output/        generated plots saved here
└── README.md

## Tools used
Python, NumPy, SciPy, Matplotlib, Plotly, HTML, JavaScript, Three.js

## What it does
generates a noisy signal, applies different filters to clean it up,
shows raw vs processed on a chart, compares which filter works best