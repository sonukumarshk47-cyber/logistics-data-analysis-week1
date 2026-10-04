# Week 2 – Data Collection, Cleaning and Preprocessing

This project demonstrates a logistics data-preprocessing pipeline using Python and Pandas.

## Objectives
- Simulate collection of logistics records.
- Identify missing values, duplicates and outliers.
- Clean and preprocess the dataset.
- Normalize numerical variables.
- Produce a cleaned dataset suitable for later analysis and modelling.

## Files
- `data/raw_logistics_data.csv` – sample dataset containing deliberate data-quality issues.
- `src/data_cleaning.py` – Python cleaning and preprocessing script.
- `data/cleaned_logistics_data.csv` – generated after running the script.
- `reports/Week_2_Logistics_Data_Cleaning_Report.docx` – complete report.

## Main techniques
1. Duplicate removal
2. Datetime conversion
3. Median imputation for missing numerical values
4. IQR-based outlier detection and capping
5. Min-Max normalization
6. Final quality validation

## Run
```bash
pip install pandas numpy scikit-learn
python src/data_cleaning.py
```

The dataset is synthetic and created for educational demonstration.
