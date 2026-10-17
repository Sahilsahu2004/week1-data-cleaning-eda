# Week 1: Data Acquisition, Cleaning and EDA (Titanic Dataset)

Virtual Data Science with Python Apprenticeship, Week 1 task.

## Overview
This project covers the first stage of a data science workflow: acquiring a public dataset, cleaning it, and exploring it with summary statistics and visualisations.

**Dataset:** Titanic passenger data (891 rows, 12 columns), loaded from a public CSV on GitHub.

## What the code does
1. Loads the data directly from a URL with pandas
2. Inspects missing values, duplicates and data types
3. Cleans the data
   - Drops Cabin (77% missing) and keeps a `HasCabin` flag
   - Fills Embarked with the mode
   - Fills Age with the median per passenger title and class
   - Converts categorical columns to the `category` type
   - Checks outliers with the IQR rule and keeps them
4. Engineers features: `Title`, `FamilySize`, `FareLog`, `AgeGroup`
5. Produces visualisations: missing-value heatmap, distributions, correlation matrix, survival by class/sex/age

## Files
- `analysis.py`: full cleaning and EDA script
- `titanic_clean.csv`: cleaned dataset
- `Week1_Data_Cleaning_EDA_Report.docx`: full written report

## How to run
```
pip install pandas numpy matplotlib seaborn
python analysis.py
```

## Key findings
- Women survived at about 74% vs 19% for men
- Survival fell from 63% (1st class) to 24% (3rd class)
- Children had the best survival rate by age group
- Small families (2-4) survived more often than solo travellers or large families
