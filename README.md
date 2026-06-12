# Digital Diffusion and the Quality–Quantity Trade-off in Women’s Labor Market Outcomes

## Evidence from a Global Panel

This repository contains the Python code used for my postgraduate thesis:

**Digital Diffusion and the Quality–Quantity Trade-off in Women’s Labor Market Outcomes: Evidence from a Global Panel**

The project investigates how digital diffusion is associated with women’s labor market outcomes across countries over time, with a particular focus on the quality–quantity trade-off in female employment.

## Project Overview

Digital technologies have reshaped labor markets around the world. However, their effects on women’s labor market outcomes may differ across countries, income groups, and institutional contexts. This project uses a global panel dataset to examine whether digital diffusion improves women’s labor market participation, job quality, or both.

The empirical analysis is organized around the following workflow:

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Baseline empirical analysis
4. Heterogeneity analysis
5. Machine learning analysis

## Repository Structure

```text
.
├── 01数据清洗.py        # Data cleaning and preprocessing
├── 02 EDA.py            # Exploratory data analysis
├── 03.py                # Baseline empirical analysis
├── 04异质性.py          # Heterogeneity analysis
├── 05机器学习.py        # Machine learning analysis
└── README.md            # Project description
```

## File Description

### 01数据清洗.py

This script is used for data cleaning and preprocessing. It prepares the raw panel data for subsequent empirical analysis.

Main tasks may include:

* Importing raw datasets
* Handling missing values
* Processing country-year panel data
* Constructing key variables
* Merging different data sources
* Exporting cleaned datasets for analysis

### 02 EDA.py

This script conducts exploratory data analysis.

Main tasks may include:

* Summary statistics
* Variable distribution analysis
* Correlation analysis
* Trend visualization
* Cross-country and time-series comparisons

This step helps provide an initial understanding of digital diffusion and women’s labor market outcomes in the global panel dataset.

### 03.py

This script contains the main empirical analysis of the thesis.

It is used to examine the relationship between digital diffusion and women’s labor market outcomes. The analysis may include baseline regression models and robustness-related checks.

Suggested file name improvement:

```text
03基准回归.py
```

or

```text
03Baseline_Analysis.py
```

### 04异质性.py

This script conducts heterogeneity analysis.

The purpose of this section is to explore whether the effects of digital diffusion differ across groups, such as:

* Different income groups
* Different regions
* Developed and developing economies
* Countries with different labor market or institutional characteristics

### 05机器学习.py

This script applies machine learning methods to further analyze or predict women’s labor market outcomes.

Possible tasks include:

* Feature selection
* Train-test split
* Model training
* Model evaluation
* Comparison of predictive performance across models
* Identification of important predictors

## Research Topic

The core research topic is the relationship between digital diffusion and women’s labor market outcomes.

In particular, the thesis focuses on the possible trade-off between:

* **Quantity of employment**: whether more women participate in the labor market or obtain employment
* **Quality of employment**: whether women obtain better, more stable, or more productive labor market opportunities

The project aims to understand whether digital diffusion contributes to improvements in both dimensions, or whether gains in one dimension may come at the expense of the other.

## Requirements

The code is written in Python. The recommended environment is:

```text
Python 3.8+
```

Commonly used Python packages may include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
```

To install common dependencies, run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

## How to Run

The scripts are intended to be run in the following order:

```bash
python 01数据清洗.py
python "02 EDA.py"
python 03.py
python 04异质性.py
python 05机器学习.py
```

Please make sure that the data files are stored in the correct local directory before running the scripts. If the scripts contain local file paths, they may need to be modified according to your own working environment.

## Notes

* This repository is mainly used for thesis code storage and replication.
* Data files may not be included due to size, privacy, or source restrictions.
* Some scripts may require modification of local file paths before running.
* Chinese file names may appear as escaped characters in Git Bash or Windows CMD. This does not affect the actual files on GitHub.

To improve filename display in Git, you may run:

```bash
git config --global core.quotepath false
```

## Author

**Houwei**

GitHub: `houwei1228`

## License

This repository is for academic and personal research purposes.
