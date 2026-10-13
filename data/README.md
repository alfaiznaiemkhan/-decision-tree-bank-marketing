Dataset

This project uses the Bank Marketing dataset from the UCI Machine Learning Repository.

Source

https://archive.ics.uci.edu/dataset/222/bank+marketing

Required File

Download and extract bank-full.csv into this directory.

Expected local path:

data/bank-full.csv

The CSV file uses a semicolon (;) delimiter.

Target Variable

The target column is y:

* yes: The customer subscribed to a term deposit.
* no: The customer did not subscribe.

Data Leakage Prevention

The duration column represents the duration of the last contact. It is excluded from the predictive features because it would not be available before a marketing call.

Reproducibility

The dataset is not committed to this repository. Download it from the official UCI source before running the project.
