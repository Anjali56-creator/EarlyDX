## Dataset inspection - diabetes

- **File:** `data/raw/diabetes/pima_indians_diabetes.csv`
- **Rows x Columns:** 768 x 9
- **Target column:** `Outcome`
- **Exact duplicate rows:** 0
- **Minority-class fraction:** 0.349

### Class distribution

| class | count |
|-------|-------|
| 0 | 500 |
| 1 | 268 |

### Missing values (NaN)

| column | n_missing |
|--------|-----------|
| Pregnancies | 0 |
| Glucose | 0 |
| BloodPressure | 0 |
| SkinThickness | 0 |
| Insulin | 0 |
| BMI | 0 |
| DiabetesPedigreeFunction | 0 |
| Age | 0 |
| Outcome | 0 |

### Zeros treated as encoded-missing

| column | n_zeros |
|--------|---------|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

### Leakage / degenerate-column check

- none flagged (no constant columns, no |corr| >= 0.95 with target)

### Summary statistics

```
                          count        mean         std     min       25%       50%        75%     max
Pregnancies               768.0    3.845052    3.369578   0.000   1.00000    3.0000    6.00000   17.00
Glucose                   768.0  120.894531   31.972618   0.000  99.00000  117.0000  140.25000  199.00
BloodPressure             768.0   69.105469   19.355807   0.000  62.00000   72.0000   80.00000  122.00
SkinThickness             768.0   20.536458   15.952218   0.000   0.00000   23.0000   32.00000   99.00
Insulin                   768.0   79.799479  115.244002   0.000   0.00000   30.5000  127.25000  846.00
BMI                       768.0   31.992578    7.884160   0.000  27.30000   32.0000   36.60000   67.10
DiabetesPedigreeFunction  768.0    0.471876    0.331329   0.078   0.24375    0.3725    0.62625    2.42
Age                       768.0   33.240885   11.760232  21.000  24.00000   29.0000   41.00000   81.00
Outcome                   768.0    0.348958    0.476951   0.000   0.00000    0.0000    1.00000    1.00
```

