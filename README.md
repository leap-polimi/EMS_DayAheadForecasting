# EMS_DayAheadForecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20109598.svg)](https://doi.org/10.5281/zenodo.20109598)

This repository contains Python scripts for forecasting the uncertain inputs required by the Energy Management System (EMS) in a day-ahead setting.

The current implementation includes workflows for:

* electrical load forecasting, contained in the `EL_LOAD` folder;
* thermal load forecasting, contained in the `TH_LOAD` folder;
* day-ahead market price forecasting, contained in the `DAM_PRICES` folder.

The same structure can be replicated for additional uncertain inputs. In particular, the `TH_LOAD` workflow can also be adapted and used for forecasting cooling load (`COOL_LOAD`), since the required workflow structure is analogous: historical dataset creation, model training, automatic day-ahead forecasting, and automatic retraining.

## Authorship and contributions

This repository is authored by:
- Andrea Scrocca
- Filippo Bovera

Affiliation: Politecnico di Milano, Department of Energy

## Citation

If you use this software in scientific work, please cite the archived release:

Scrocca, A., & Bovera, F. (2026). *EMS_DayAheadForecasting* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20109598

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff).

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

---

## Repository structure

```text
EMS_DayAheadForecasting/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- EL_LOAD/
|   |-- .env.example
|   |-- 0_EL_LOAD_creation_dataset.py
|   |-- 1_EL_LOAD_model_training.py
|   |-- 2_EL_LOAD_automatic_forecasting.py
|   `-- 2_EL_LOAD_automatic_retraining.py
|
|-- TH_LOAD/
|   |-- .env.example
|   |-- 0_TH_LOAD_creation_dataset.py
|   |-- 1_TH_LOAD_model_training.py
|   |-- 2_TH_LOAD_automatic_forecasting.py
|   `-- 2_TH_LOAD_automatic_retraining.py
|
`-- DAM_PRICES/
    |-- .env.example
    |-- 2_DAM_PRICES_automatic_forecasting_retraining.py
    |-- PUN_dataset_final.csv
    `-- NORD_dataset_final.csv
```

Each workflow requires a private `.env` file to run locally. The `.env` files contain private credentials and local configuration values and must **never** be uploaded to GitHub.

---

# Electrical Load Forecasting (`EL_LOAD`)

The `EL_LOAD` folder contains the full workflow used to create, train, run, and update day-ahead electrical load forecasting models for the EMS.

This workflow exploits the interaction with the **Optimo Cloud**, which stores and provides access to measurement data from **PoliGrid**, the Multi-Energy Microgrid (MEMG) infrastructure at Politecnico di Milano [(link)](https://www.polimi.it/en/sustainable-development/environment/energy-and-decarbonisation/energy-committee/translate-to-english-poligrid). The cloud infrastructure and API access are developed by [**Optimo IoT**](https://optimoiot.it/).

In addition to measurement data from Optimo Cloud, the workflow also interacts with [**Open-Meteo**](https://open-meteo.com/) to retrieve weather data. The weather variables currently used include outdoor temperature, relative humidity, dew point, and direct normal irradiance.

The workflow is organized into four scripts:

```text
EL_LOAD/
|-- .env.example
|-- 0_EL_LOAD_creation_dataset.py
|-- 1_EL_LOAD_model_training.py
|-- 2_EL_LOAD_automatic_forecasting.py
`-- 2_EL_LOAD_automatic_retraining.py
```

---

## 1. `0_EL_LOAD_creation_dataset.py`

This script creates the historical dataset used to train the electrical load forecasting models.

It downloads and processes:

* electrical consumption measurements from Optimo Cloud;
* PV production measurements;
* CHP and main grid exchange measurements;
* weather data from Open-Meteo;
* calendar features, such as month, weekday, quarter-hour, holidays, and building opening/closing status.

The final output is an Excel dataset, usually saved as:

```text
EL_LOAD/dataset.xlsx
```

Typical command:

```bash
python EL_LOAD/0_EL_LOAD_creation_dataset.py
```

Example with custom dates:

```bash
python EL_LOAD/0_EL_LOAD_creation_dataset.py --start-date 2024-05-30 --end-date 2025-09-30
```

Example using today's date as the end date:

```bash
python EL_LOAD/0_EL_LOAD_creation_dataset.py --start-date 2024-05-30 --end-date today
```

---

## 2. `1_EL_LOAD_model_training.py`

This script trains one Random Forest model for each electrical load target.

The targets include:

* gross electrical consumption of each cabin;
* total gross electrical consumption;
* total net electrical consumption.

The script:

1. loads the dataset created by `0_EL_LOAD_creation_dataset.py`;
2. removes known bad-data periods;
3. drops rows with missing values;
4. creates lagged target features, for example D-2 to D-7;
5. performs hyperparameter tuning using `GridSearchCV`;
6. evaluates each model on a test period;
7. saves the trained models as `.joblib` files;
8. saves a metrics summary file.

The outputs are usually saved in:

```text
EL_LOAD/models/
EL_LOAD/results/metrics_summary.xlsx
```

Typical command:

```bash
python EL_LOAD/1_EL_LOAD_model_training.py
```

Example with custom train/test split:

```bash
python EL_LOAD/1_EL_LOAD_model_training.py --min-date 2024-06-08T08:00:00 --split-date 2025-06-08T23:59:00
```

---

## 3. `2_EL_LOAD_automatic_forecasting.py`

This script runs the day-ahead electrical load forecast.

It is intended to be executed automatically every day, for example at 10:00.

The script:

1. defines the forecast day, by default tomorrow in `Europe/Rome` time;
2. downloads the weather forecast from Open-Meteo;
3. downloads recent historical electrical data from Optimo Cloud;
4. builds the same features used during training;
5. loads the trained `.joblib` models from `EL_LOAD/models/`;
6. generates 15-minute forecasts for the next day;
7. saves the forecast to Excel;
8. optionally uploads selected forecast outputs to Optimo Cloud.

The output is usually saved in:

```text
EL_LOAD/forecasts/
```

Typical command:

```bash
python EL_LOAD/2_EL_LOAD_automatic_forecasting.py
```

Forecast a specific date:

```bash
python EL_LOAD/2_EL_LOAD_automatic_forecasting.py --forecast-date 2025-10-01
```

Run locally without uploading results to Optimo:

```bash
python EL_LOAD/2_EL_LOAD_automatic_forecasting.py --no-upload
```

This is the recommended command for testing.

---

## 4. `2_EL_LOAD_automatic_retraining.py`

This script updates the electrical load dataset and retrains the forecasting models.

It is intended to be executed periodically, for example once per month.

The script:

1. loads the existing dataset;
2. appends new data up to the latest reliable historical weather date;
3. removes known bad-data periods;
4. keeps the most recent training window, for example the last 365 days;
5. loads the best hyperparameters found during the initial training;
6. retrains all Random Forest models;
7. overwrites the `.joblib` model files in `EL_LOAD/models/`.

Typical command:

```bash
python EL_LOAD/2_EL_LOAD_automatic_retraining.py
```

Run retraining without appending new data:

```bash
python EL_LOAD/2_EL_LOAD_automatic_retraining.py --skip-dataset-update
```

This is useful for testing the retraining logic on an existing dataset.

---

# Thermal Load Forecasting (`TH_LOAD`)

The `TH_LOAD` folder contains the full workflow used to create, train, run, and update day-ahead thermal load forecasting models for the EMS.

As for the electrical load workflow, the thermal workflow exploits the interaction with **Optimo Cloud** to retrieve measured thermal load data from the PoliGrid infrastructure at Politecnico di Milano. It also interacts with **Open-Meteo** to retrieve historical weather data for dataset creation and retraining, and weather forecasts for daily day-ahead prediction.

The same workflow can also be adapted for cooling load forecasting (`COOL_LOAD`). In that case, the thermal load target variables and measurement identifiers should be replaced by the corresponding cooling load measurements, while keeping the same overall structure.

The thermal load workflow currently considers two forecasting targets:

* `THERMAL_LOAD_kW`: total campus thermal load, computed as the sum of the thermal loads measured at the campus substations;
* `DH_THERMAL_LOAD_kW`: district-heating thermal load, derived from measurements collected at the inlet and outlet of the district heating network.

The workflow is organized into four scripts:

```text
TH_LOAD/
|-- .env.example
|-- 0_TH_LOAD_creation_dataset.py
|-- 1_TH_LOAD_model_training.py
|-- 2_TH_LOAD_automatic_forecasting.py
`-- 2_TH_LOAD_automatic_retraining.py
```

---

## 1. `0_TH_LOAD_creation_dataset.py`

This script creates the historical dataset used to train the thermal load forecasting models.

It downloads and processes:

* measured thermal load data from Optimo Cloud;
* measured district-heating thermal load data from Optimo Cloud;
* weather data from Open-Meteo;
* calendar features, such as month, weekday, quarter-hour, holidays, and building opening/closing status;
* derived features such as `T_open`, which combines outdoor temperature with the building opening/closing status.

The final output is an Excel dataset, usually saved as:

```text
TH_LOAD/dataset_thermal.xlsx
```

Typical command:

```bash
python TH_LOAD/0_TH_LOAD_creation_dataset.py
```

Example with custom dates:

```bash
python TH_LOAD/0_TH_LOAD_creation_dataset.py --start-date 2024-12-03 --end-date 2025-09-30
```

Example using today's date as the end date:

```bash
python TH_LOAD/0_TH_LOAD_creation_dataset.py --start-date 2024-12-03 --end-date today
```

---

## 2. `1_TH_LOAD_model_training.py`

This script trains Random Forest models for the thermal load targets.

The targets currently include:

* `THERMAL_LOAD_kW`;
* `DH_THERMAL_LOAD_kW`.

The script:

1. loads the dataset created by `0_TH_LOAD_creation_dataset.py`;
2. sorts and reindexes the data on a continuous 15-minute grid;
3. creates lagged target features, for example D-2 to D-7;
4. applies target-specific filtering rules, such as removing missing values and zero-load periods when required;
5. performs hyperparameter tuning using Random Forest models;
6. evaluates the models on a test period;
7. saves trained models as `.joblib` files;
8. saves metrics, feature importance files, and optional verification outputs.

The outputs are usually saved in:

```text
TH_LOAD/models/
TH_LOAD/results/metrics_summary_thermal.xlsx
```

Typical command:

```bash
python TH_LOAD/1_TH_LOAD_model_training.py
```

Example with custom train/test dates:

```bash
python TH_LOAD/1_TH_LOAD_model_training.py --train-end 2025-04-30T23:59:00 --test-start 2025-05-01T00:00:00
```

---

## 3. `2_TH_LOAD_automatic_forecasting.py`

This script runs the day-ahead thermal load forecast.

It is intended to be executed automatically every day, for example at 10:00.

The script:

1. defines the forecast day, by default tomorrow in `Europe/Rome` time;
2. downloads the weather forecast from Open-Meteo;
3. downloads recent historical thermal load data from Optimo Cloud;
4. builds the same features used during training;
5. loads the trained `.joblib` models from `TH_LOAD/models/`;
6. generates 15-minute forecasts for the next day;
7. saves the forecast files to Excel;
8. optionally uploads selected forecast outputs to Optimo Cloud.

The output is usually saved in:

```text
TH_LOAD/forecasts/
```

Typical command:

```bash
python TH_LOAD/2_TH_LOAD_automatic_forecasting.py
```

Forecast a specific date:

```bash
python TH_LOAD/2_TH_LOAD_automatic_forecasting.py --forecast-date 2025-10-01
```

Run locally without uploading results to Optimo:

```bash
python TH_LOAD/2_TH_LOAD_automatic_forecasting.py --no-upload
```

This is the recommended command for testing.

---

## 4. `2_TH_LOAD_automatic_retraining.py`

This script updates the thermal load dataset and retrains the forecasting models.

It is intended to be executed periodically, for example once per month.

The script:

1. loads the existing thermal dataset;
2. appends recent thermal load and weather data;
3. applies the same cleaning and feature engineering rules used during training;
4. keeps the most recent training window, for example the last 365 days;
5. loads the best hyperparameters found during the initial training;
6. retrains all thermal Random Forest models;
7. overwrites the `.joblib` model files in `TH_LOAD/models/`.

Typical command:

```bash
python TH_LOAD/2_TH_LOAD_automatic_retraining.py
```

Run retraining without appending new data:

```bash
python TH_LOAD/2_TH_LOAD_automatic_retraining.py --skip-dataset-update
```

This is useful for testing the retraining logic on an existing dataset.

---

# Day-Ahead Market Price Forecasting (`DAM_PRICES`)

The `DAM_PRICES` folder contains the workflow used to forecast day-ahead market prices for the EMS.

This workflow uses the [**GME API**](https://www.mercatoelettrico.org/) to retrieve market price data from the Italian electricity market. The forecasting model is based on the LEAR approach and produces hourly day-ahead forecasts for:

* `PUN`: the Italian national single price;
* `NORD`: the northern Italian market zone price.

The DAM price workflow assumes that the historical datasets are already available in the `DAM_PRICES` folder with the following names:

```text
DAM_PRICES/PUN_dataset_final.csv
DAM_PRICES/NORD_dataset_final.csv
```

These CSV files must contain at least the following columns:

```text
italian datetime
datetime
price
```

where `datetime` is the UTC timestamp and `price` is the historical market price in EUR/MWh.

The workflow is organized as follows:

```text
DAM_PRICES/
|-- .env.example
|-- 2_DAM_PRICES_automatic_forecasting_retraining.py
|-- PUN_dataset_final.csv
`-- NORD_dataset_final.csv
```

---

## `2_DAM_PRICES_automatic_forecasting_retraining.py`

This script runs the automatic day-ahead price forecasting workflow.

It is intended to be executed automatically every day, for example after the day-ahead market results are available.

The script:

1. checks the latest market dates available in the PUN and NORD CSV datasets;
2. downloads from the GME API only the missing market dates required for the forecast;
3. appends the missing data to the historical CSV datasets;
4. runs the LEAR model to forecast the selected day-ahead market prices;
5. saves the forecast files locally;
6. optionally uploads selected forecast outputs to Optimo Cloud.

If the PUN and NORD datasets already contain the market data needed for the forecast, the script skips the GME download step.

The output is usually saved in:

```text
DAM_PRICES/forecasts/
```

Typical command:

```bash
python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py
```

Forecast a specific date:

```bash
python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --forecast-date 2026-05-11
```

Run locally without uploading results to Optimo:

```bash
python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --no-upload
```

Force the script to skip the GME update and use only the existing CSV datasets:

```bash
python DAM_PRICES/2_DAM_PRICES_automatic_forecasting_retraining.py --no-upload --skip-gme-update
```

This is useful for testing or when the required market data are already present in the local datasets.

---

# Required local configuration

Each user must create a private `.env` file inside each workflow folder that they want to run.

For the electrical load workflow:

```text
EL_LOAD/.env
```

For the thermal load workflow:

```text
TH_LOAD/.env
```

For the day-ahead market price workflow:

```text
DAM_PRICES/.env
```

These files contain private credentials and local configuration values. They must **never** be committed to GitHub.

Public templates are provided as:

```text
EL_LOAD/.env.example
TH_LOAD/.env.example
DAM_PRICES/.env.example
```