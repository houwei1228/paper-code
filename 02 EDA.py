import pandas as pd
import numpy as np


DATA_PATH = r"D:/课程/论文/投稿论文/数据/panel_before_imputation_2006_2023.csv"
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]


# secure_employment = 100 - female_vulnerable_emp
if "secure_employment" not in df.columns and "female_vulnerable_emp" in df.columns:
    df["secure_employment"] = 100 - df["female_vulnerable_emp"]

# log_gdp_pc_ppp = ln(gdp_pc_ppp)
if "log_gdp_pc_ppp" not in df.columns and "gdp_pc_ppp" in df.columns:
    df["log_gdp_pc_ppp"] = np.where(df["gdp_pc_ppp"] > 0, np.log(df["gdp_pc_ppp"]), np.nan)


vars_for_desc = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment",
    "internet_use",
    "log_gdp_pc_ppp",
    "fertility",
    "urbanization",
    "service_share"
]

missing_vars = [v for v in vars_for_desc if v not in df.columns]
if missing_vars:
    raise ValueError(f"以下变量不存在，请先检查数据或变量名：{missing_vars}")


desc_sample = df.dropna(subset=vars_for_desc).copy()

print("Descriptive-statistics sample shape:", desc_sample.shape)
print("Number of countries:", desc_sample["iso3"].nunique() if "iso3" in desc_sample.columns else "N/A")
print("Year range:", desc_sample["year"].min(), "-", desc_sample["year"].max() if "year" in desc_sample.columns else "N/A")


var_labels = {
    "secure_employment": "Secure employment",
    "female_vulnerable_emp": "Female vulnerable employment",
    "female_lfp": "Female labor force participation",
    "female_employment_rate": "Female employment rate",
    "female_unemployment": "Female unemployment",
    "internet_use": "Internet use",
    "log_gdp_pc_ppp": "Log GDP per capita (PPP)",
    "fertility": "Fertility rate",
    "urbanization": "Urbanization",
    "service_share": "Service sector share"
}


rows = []

for v in vars_for_desc:
    s = desc_sample[v].dropna()
    rows.append({
        "Variable": var_labels.get(v, v),
        "N": s.shape[0],
        "Mean": s.mean(),
        "SD": s.std(),
        "Min": s.min(),
        "P25": s.quantile(0.25),
        "Median": s.median(),
        "P75": s.quantile(0.75),
        "Max": s.max()
    })

desc_table = pd.DataFrame(rows)


for col in ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]:
    desc_table[col] = desc_table[col].round(3)

print("\nTable 4.X Descriptive statistics:\n")
print(desc_table.to_string(index=False))


OUTPUT_PATH = r"D:/课程/论文/投稿论文/数据/table_4x_descriptive_statistics.csv"
desc_table.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nDescriptive statistics table saved to:\n{OUTPUT_PATH}")