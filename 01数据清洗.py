import pandas as pd
import numpy as np
from datetime import datetime


DATA_PATH = r"D:/课程/论文/投稿论文/数据/总数据.csv"

PANEL_RAW_PATH = r"D:/课程/论文/投稿论文/数据/panel_raw_2006_2023.csv"
PANEL_BEFORE_PATH = r"D:/课程/论文/投稿论文/数据/panel_before_imputation_2006_2023.csv"
PANEL_AFTER_PATH = r"D:/课程/论文/投稿论文/数据/panel_after_imputation_2006_2023.csv"



def safe_to_csv(df: pd.DataFrame, path: str, label: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{label}已保存：{path}")
    except PermissionError:
        alt_path = path.replace(".csv", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(alt_path, index=False, encoding="utf-8-sig")
        print(f"{label}目标文件被占用，已另存为：{alt_path}")



df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = [c.strip() for c in df_raw.columns]

required_cols = ["Country Name", "Country Code", "Series Name", "Series Code"]
for col in required_cols:
    if col not in df_raw.columns:
        raise ValueError(f"缺少必要列: {col}")

year_cols = [f"{y} [YR{y}]" for y in range(2006, 2024)]
missing_year_cols = [c for c in year_cols if c not in df_raw.columns]
if missing_year_cols:
    raise ValueError(f"缺少年份列: {missing_year_cols}")


#  删除非主权/聚合体样本

aggregate_codes = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC",
    "WLD"
}

non_sovereign_codes = {
    "ABW","ASM","BMU","CYM","CUW","FRO","GIB","GRL","GUM","HKG","IMN","KNA",
    "LIE","MAC","MNP","NCL","PRI","PSE","PYF","TCA","VGB","VIR","XKX",
    "AND","MCO","SMR"
}

aggregate_name_keywords = [
    "World",
    "income",
    "Euro area",
    "European Union",
    "Sub-Saharan Africa",
    "Middle East",
    "North America",
    "Latin America",
    "Caribbean",
    "South Asia",
    "East Asia",
    "Pacific",
    "Arab World",
    "fragile",
    "IBRD",
    "IDA",
    "OECD",
    "members",
    "small states",
    "Heavily indebted",
    "least developed",
    "Early-demographic",
    "Late-demographic",
    "Post-demographic",
    "Pre-demographic"
]

sample_base = (
    df_raw[["Country Name", "Country Code"]]
    .drop_duplicates()
    .rename(columns={"Country Name": "country_name", "Country Code": "iso3"})
)

def is_valid_country(row):
    name = str(row["country_name"]).strip()
    code = str(row["iso3"]).strip()

    if code in aggregate_codes:
        return False
    if code in non_sovereign_codes:
        return False
    if len(code) != 3:
        return False

    lower_name = name.lower()
    for kw in aggregate_name_keywords:
        if kw.lower() in lower_name:
            return False
    return True

sample_base["keep"] = sample_base.apply(is_valid_country, axis=1)
keep_iso3 = set(sample_base.loc[sample_base["keep"], "iso3"])


#  变量重命名与标准化

series_map = {
    "IT.NET.USER.ZS": "internet_use",
    "IT.NET.USER.FE.ZS": "internet_female_use",
    "IT.NET.USER.MA.ZS": "internet_male_use",
    "SL.TLF.CACT.FE.ZS": "female_lfp",
    "SL.EMP.VULN.FE.ZS": "female_vulnerable_emp",
    "SL.EMP.TOTL.SP.FE.ZS": "female_employment_rate",
    "SL.UEM.TOTL.FE.ZS": "female_unemployment",
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
    "SP.DYN.TFRT.IN": "fertility",
    "SP.URB.TOTL.IN.ZS": "urbanization",
    "NV.SRV.TOTL.ZS": "service_share",
    "SE.TER.ENRR.FE": "female_tertiary_enroll",
    "SE.TER.ENRR.MA": "male_tertiary_enroll",
    "NE.TRD.GNFS.ZS": "trade_openness",
    "SG.LAW.INDX": "wbl_index",
    "IT.NET.BBND.P2": "fixed_broadband",
    "IT.CEL.SETS.P2": "mobile_subscriptions"
}

df = df_raw.copy()
df = df[df["Country Code"].isin(keep_iso3)].copy()
df = df[df["Series Code"].isin(series_map.keys())].copy()

df = df.rename(columns={
    "Country Name": "country_name",
    "Country Code": "iso3",
    "Series Name": "series_name",
    "Series Code": "series_code"
})

df["var_name"] = df["series_code"].map(series_map)


#转为面板数据

df_long = df.melt(
    id_vars=["country_name", "iso3", "series_name", "series_code", "var_name"],
    value_vars=year_cols,
    var_name="year_raw",
    value_name="value"
)

df_long["year"] = df_long["year_raw"].str.extract(r"(\d{4})").astype(int)
df_long["value"] = df_long["value"].replace("..", np.nan).replace("", np.nan)
df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

panel = df_long.pivot_table(
    index=["country_name", "iso3", "year"],
    columns="var_name",
    values="value",
    aggfunc="first"
).reset_index()

panel.columns.name = None
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)



panel_raw = panel.copy()

print("\n================ panel_raw 概览 ================\n")
print("panel_raw 维度：", panel_raw.shape)
print("panel_raw 列数：", len(panel_raw.columns))

safe_to_csv(panel_raw, PANEL_RAW_PATH, "panel_raw")


# panel_before：插补前分析表

panel_before = panel_raw.copy()


if "trade_openness" in panel_before.columns and "trade_open" not in panel_before.columns:
    panel_before["trade_open"] = panel_before["trade_openness"]

if "female_tertiary_enroll" in panel_before.columns and "female_tertiary" not in panel_before.columns:
    panel_before["female_tertiary"] = panel_before["female_tertiary_enroll"]

if "male_tertiary_enroll" in panel_before.columns and "male_tertiary" not in panel_before.columns:
    panel_before["male_tertiary"] = panel_before["male_tertiary_enroll"]


if {"internet_male_use", "internet_female_use"}.issubset(panel_before.columns):
    panel_before["digital_gender_gap_abs"] = panel_before["internet_male_use"] - panel_before["internet_female_use"]
    panel_before["digital_gender_gap"] = np.where(
        panel_before["internet_male_use"] > 0,
        1 - panel_before["internet_female_use"] / panel_before["internet_male_use"],
        np.nan
    )

if {"female_tertiary_enroll", "male_tertiary_enroll"}.issubset(panel_before.columns):
    panel_before["tertiary_gender_gap"] = panel_before["female_tertiary_enroll"] - panel_before["male_tertiary_enroll"]

if "gdp_pc_ppp" in panel_before.columns:
    panel_before["log_gdp_pc_ppp"] = np.where(panel_before["gdp_pc_ppp"] > 0, np.log(panel_before["gdp_pc_ppp"]), np.nan)

if "female_vulnerable_emp" in panel_before.columns:
    panel_before["secure_employment"] = 100 - panel_before["female_vulnerable_emp"]

if "female_unemployment" in panel_before.columns:
    panel_before["employment_security_unemp"] = -panel_before["female_unemployment"]

if {"female_tertiary", "male_tertiary"}.issubset(panel_before.columns):
    panel_before["female_edu_ratio"] = np.where(
        panel_before["male_tertiary"] > 0,
        panel_before["female_tertiary"] / panel_before["male_tertiary"],
        np.nan
    )
    panel_before["female_edu_gap"] = np.where(
        panel_before["male_tertiary"] > 0,
        1 - panel_before["female_tertiary"] / panel_before["male_tertiary"],
        np.nan
    )

id_cols = ["country_name", "iso3", "year"]
panel_vars = [c for c in panel_before.columns if c not in id_cols]

missing_total = pd.DataFrame({
    "variable": panel_vars,
    "non_missing_n": [panel_before[v].notna().sum() for v in panel_vars],
    "missing_n": [panel_before[v].isna().sum() for v in panel_vars],
    "missing_rate": [panel_before[v].isna().mean() for v in panel_vars]
}).sort_values("missing_rate", ascending=False)

print("\n================ 5.1 总缺失率表 ================\n")
print(missing_total.to_string(index=False))


vars_A = [
    "internet_use",
    "female_lfp",
    "female_vulnerable_emp",
    "gdp_pc_ppp",
    "fertility",
    "urbanization",
    "service_share"
]

vars_B = vars_A + ["trade_open", "wbl_index"]

vars_C = vars_B + [
    "female_tertiary",
    "internet_female_use",
    "internet_male_use",
    "digital_gender_gap"
]

vars_A_exist = [v for v in vars_A if v in panel_before.columns]
vars_B_exist = [v for v in vars_B if v in panel_before.columns]
vars_C_exist = [v for v in vars_C if v in panel_before.columns]

panel_before["sample_A_complete"] = panel_before[vars_A_exist].notna().all(axis=1).astype(int)
panel_before["sample_B_complete"] = panel_before[vars_B_exist].notna().all(axis=1).astype(int)
panel_before["sample_C_complete"] = panel_before[vars_C_exist].notna().all(axis=1).astype(int)
panel_before["sample_A_candidate"] = panel_before[["internet_use", "female_lfp"]].notna().all(axis=1).astype(int)

# 删除 A 样本 0 年有效观测国家

a_years_by_country = (
    panel_before.groupby(["country_name", "iso3"])["sample_A_complete"]
    .sum()
    .reset_index(name="n_years_A")
)

drop_iso3_A0 = set(a_years_by_country.loc[a_years_by_country["n_years_A"] == 0, "iso3"])

print("\n================ 删除前：A样本0年国家 ================\n")
print("A样本0年国家数：", len(drop_iso3_A0))
if len(drop_iso3_A0) > 0:
    print(a_years_by_country.loc[a_years_by_country["n_years_A"] == 0].to_string(index=False))


panel_raw = panel_raw.loc[~panel_raw["iso3"].isin(drop_iso3_A0)].copy()
panel_before = panel_before.loc[~panel_before["iso3"].isin(drop_iso3_A0)].copy()

print("\n删除 A=0 年国家后：")
print("panel_raw 国家数：", panel_raw["iso3"].nunique())
print("panel_before 国家数：", panel_before["iso3"].nunique())

sample_summary = pd.DataFrame({
    "dataset": ["A_main", "B_extended_controls", "C_extended_mechanism"],
    "n_vars_required": [len(vars_A_exist), len(vars_B_exist), len(vars_C_exist)],
    "n_obs_complete": [
        panel_before["sample_A_complete"].sum(),
        panel_before["sample_B_complete"].sum(),
        panel_before["sample_C_complete"].sum()
    ],
    "n_country_complete": [
        panel_before.loc[panel_before["sample_A_complete"] == 1, "iso3"].nunique(),
        panel_before.loc[panel_before["sample_B_complete"] == 1, "iso3"].nunique(),
        panel_before.loc[panel_before["sample_C_complete"] == 1, "iso3"].nunique()
    ],
    "avg_years_per_country": [
        panel_before.loc[panel_before["sample_A_complete"] == 1].groupby("iso3").size().mean(),
        panel_before.loc[panel_before["sample_B_complete"] == 1].groupby("iso3").size().mean(),
        panel_before.loc[panel_before["sample_C_complete"] == 1].groupby("iso3").size().mean()
    ]
})

print("\n================ 第6步：模型分层样本概览 ================\n")
print(sample_summary.to_string(index=False))

# 异常值检查与 winsorize
outlier_vars = ["gdp_pc_ppp", "mobile_subscriptions", "fixed_broadband", "trade_open"]
outlier_vars = [v for v in outlier_vars if v in panel_before.columns]

def quantile_summary(series):
    s = series.dropna()
    if len(s) == 0:
        return pd.Series({
            "n": 0, "min": np.nan, "p1": np.nan, "p5": np.nan,
            "p25": np.nan, "p50": np.nan, "p75": np.nan,
            "p95": np.nan, "p99": np.nan, "max": np.nan
        })
    return pd.Series({
        "n": s.shape[0],
        "min": s.min(),
        "p1": s.quantile(0.01),
        "p5": s.quantile(0.05),
        "p25": s.quantile(0.25),
        "p50": s.quantile(0.50),
        "p75": s.quantile(0.75),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
        "max": s.max()
    })

outlier_summary_list = []
for v in outlier_vars:
    temp = quantile_summary(panel_before[v])
    temp["variable"] = v
    outlier_summary_list.append(temp)

outlier_summary = pd.DataFrame(outlier_summary_list)
if not outlier_summary.empty:
    outlier_summary = outlier_summary[
        ["variable", "n", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]
    ]
    print("\n================ 第7步：异常值分位数检查 ================\n")
    print(outlier_summary.to_string(index=False))

def winsorize_series(s, lower=0.01, upper=0.99):
    x = s.copy()
    lo = x.quantile(lower)
    hi = x.quantile(upper)
    return x.clip(lower=lo, upper=hi)

winsor_vars = ["gdp_pc_ppp", "trade_open", "mobile_subscriptions", "fixed_broadband"]
winsor_vars = [v for v in winsor_vars if v in panel_before.columns]

for v in winsor_vars:
    panel_before[f"{v}_w1"] = winsorize_series(panel_before[v], 0.01, 0.99)
    panel_before[f"{v}_w25"] = winsorize_series(panel_before[v], 0.025, 0.975)


if "gdp_pc_ppp_w1" in panel_before.columns:
    panel_before["log_gdp_pc_ppp_w1"] = np.where(panel_before["gdp_pc_ppp_w1"] > 0, np.log(panel_before["gdp_pc_ppp_w1"]), np.nan)

if "fixed_broadband" in panel_before.columns:
    panel_before["log_fixed_broadband"] = np.log1p(panel_before["fixed_broadband"])

if "fixed_broadband_w1" in panel_before.columns:
    panel_before["log_fixed_broadband_w1"] = np.log1p(panel_before["fixed_broadband_w1"])

if "mobile_subscriptions" in panel_before.columns:
    panel_before["log_mobile_subscriptions"] = np.log1p(panel_before["mobile_subscriptions"])

if "mobile_subscriptions_w1" in panel_before.columns:
    panel_before["log_mobile_subscriptions_w1"] = np.log1p(panel_before["mobile_subscriptions_w1"])

print("\n================ panel_before 概览 ================\n")
print("panel_before 维度：", panel_before.shape)
print("panel_before 列数：", len(panel_before.columns))

safe_to_csv(panel_before, PANEL_BEFORE_PATH, "panel_before")


#  panel_after：插补后分析
panel_after = panel_before.copy()
panel_after = panel_after.sort_values(["iso3", "year"]).reset_index(drop=True)

low_missing_vars = [
    "gdp_pc_ppp",
    "internet_use",
    "wbl_index",
    "fixed_broadband",
    "mobile_subscriptions",
    "female_employment_rate",
    "female_lfp",
    "female_unemployment",
    "female_vulnerable_emp",
    "secure_employment",
    "employment_security_unemp"
]
low_missing_vars = [v for v in low_missing_vars if v in panel_after.columns]

mid_missing_vars = [
    "trade_open",
    "female_tertiary",
    "male_tertiary"
]
mid_missing_vars = [v for v in mid_missing_vars if v in panel_after.columns]

high_missing_vars = [
    "internet_female_use",
    "internet_male_use",
    "digital_gender_gap"
]
high_missing_vars = [v for v in high_missing_vars if v in panel_after.columns]

def fill_edges_one_year_only(series: pd.Series) -> pd.Series:
    s = series.copy()
    if len(s) >= 2 and pd.isna(s.iloc[0]) and pd.notna(s.iloc[1]):
        s.iloc[0] = s.iloc[1]
    if len(s) >= 2 and pd.isna(s.iloc[-1]) and pd.notna(s.iloc[-2]):
        s.iloc[-1] = s.iloc[-2]
    return s

def interpolate_internal_with_gap_limit(series: pd.Series, max_gap: int = 2) -> pd.Series:
    s = series.copy()
    return s.interpolate(method="linear", limit=max_gap, limit_area="inside")

def impute_by_country_basic(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    var_list: list,
    max_gap: int = 2,
    edge_fill_one_year: bool = True,
    skip_all_missing_country: bool = False
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    for var in var_list:
        imp_col = f"{var}_imp"
        flag_col = f"{var}_imputed_flag"

        out[imp_col] = out[var].copy()
        out[flag_col] = 0

        pieces = []

        for _, sub in out.groupby(group_col, sort=False):
            sub = sub.sort_values(time_col).copy()
            original = sub[var].copy()

            if skip_all_missing_country and original.notna().sum() == 0:
                sub[imp_col] = original
                sub[flag_col] = 0
                pieces.append(sub)
                continue

            imputed = interpolate_internal_with_gap_limit(original, max_gap=max_gap)

            if edge_fill_one_year:
                imputed = fill_edges_one_year_only(imputed)

            flag = original.isna() & imputed.notna()

            sub[imp_col] = imputed
            sub[flag_col] = flag.astype(int)
            pieces.append(sub)

        out = pd.concat(pieces, axis=0).sort_values([group_col, time_col]).reset_index(drop=True)

    return out

panel_after = impute_by_country_basic(
    df=panel_after,
    group_col="iso3",
    time_col="year",
    var_list=low_missing_vars,
    max_gap=2,
    edge_fill_one_year=True,
    skip_all_missing_country=False
)

panel_after = impute_by_country_basic(
    df=panel_after,
    group_col="iso3",
    time_col="year",
    var_list=mid_missing_vars,
    max_gap=2,
    edge_fill_one_year=True,
    skip_all_missing_country=True
)

for var in high_missing_vars:
    panel_after[f"{var}_imp"] = panel_after[var]
    panel_after[f"{var}_imputed_flag"] = 0

imputation_summary = []
all_impute_vars = low_missing_vars + mid_missing_vars + high_missing_vars

for var in all_impute_vars:
    imp_col = f"{var}_imp"
    flag_col = f"{var}_imputed_flag"
    if imp_col in panel_after.columns:
        imputation_summary.append({
            "variable": var,
            "original_non_missing": int(panel_after[var].notna().sum()),
            "original_missing": int(panel_after[var].isna().sum()),
            "imputed_non_missing": int(panel_after[imp_col].notna().sum()),
            "remaining_missing": int(panel_after[imp_col].isna().sum()),
            "n_imputed_cells": int(panel_after[flag_col].sum())
        })

imputation_summary = pd.DataFrame(imputation_summary)
print("\n================ 插补结果汇总 ================\n")
print(imputation_summary.to_string(index=False))


analysis_vars_to_replace = low_missing_vars + mid_missing_vars
analysis_vars_to_replace = [v for v in analysis_vars_to_replace if f"{v}_imp" in panel_after.columns]

for var in analysis_vars_to_replace:
    panel_after[f"{var}_analysis"] = panel_after[f"{var}_imp"]

for var in high_missing_vars:
    if var in panel_after.columns:
        panel_after[f"{var}_analysis"] = panel_after[var]


if "gdp_pc_ppp_analysis" in panel_after.columns:
    panel_after["log_gdp_pc_ppp_analysis"] = np.where(
        panel_after["gdp_pc_ppp_analysis"] > 0,
        np.log(panel_after["gdp_pc_ppp_analysis"]),
        np.nan
    )

if "trade_open_analysis" in panel_after.columns and "trade_openness_analysis" not in panel_after.columns:
    panel_after["trade_openness_analysis"] = panel_after["trade_open_analysis"]

if "wbl_index_imp" in panel_after.columns:
    panel_after["wbl_index_analysis"] = panel_after["wbl_index_imp"]

if "female_lfp_imp" in panel_after.columns:
    panel_after["female_lfp_analysis"] = panel_after["female_lfp_imp"]

if "female_vulnerable_emp_imp" in panel_after.columns:
    panel_after["female_vulnerable_emp_analysis"] = panel_after["female_vulnerable_emp_imp"]

if "female_employment_rate_imp" in panel_after.columns:
    panel_after["female_employment_rate_analysis"] = panel_after["female_employment_rate_imp"]

if "female_unemployment_imp" in panel_after.columns:
    panel_after["female_unemployment_analysis"] = panel_after["female_unemployment_imp"]

if "internet_use_imp" in panel_after.columns:
    panel_after["internet_use_analysis"] = panel_after["internet_use_imp"]

if "female_tertiary_imp" in panel_after.columns:
    panel_after["female_tertiary_analysis"] = panel_after["female_tertiary_imp"]

if "male_tertiary_imp" in panel_after.columns:
    panel_after["male_tertiary_analysis"] = panel_after["male_tertiary_imp"]

if "fixed_broadband_imp" in panel_after.columns:
    panel_after["fixed_broadband_analysis"] = panel_after["fixed_broadband_imp"]

if "mobile_subscriptions_imp" in panel_after.columns:
    panel_after["mobile_subscriptions_analysis"] = panel_after["mobile_subscriptions_imp"]

if "female_vulnerable_emp_analysis" in panel_after.columns:
    panel_after["secure_employment_analysis"] = 100 -panel_after["female_vulnerable_emp_analysis"]

if "female_tertiary_analysis" in panel_after.columns and "male_tertiary_analysis" in panel_after.columns:
    panel_after["female_edu_ratio_analysis"] = np.where(
        panel_after["male_tertiary_analysis"] > 0,
        panel_after["female_tertiary_analysis"] / panel_after["male_tertiary_analysis"],
        np.nan
    )
    panel_after["female_edu_gap_analysis"] = np.where(
        panel_after["male_tertiary_analysis"] > 0,
        1 - panel_after["female_tertiary_analysis"] / panel_after["male_tertiary_analysis"],
        np.nan
    )

if "internet_female_use_analysis" in panel_after.columns and "internet_male_use_analysis" in panel_after.columns:
    panel_after["digital_gender_gap_abs_analysis"] = (
        panel_after["internet_male_use_analysis"] - panel_after["internet_female_use_analysis"]
    )
    panel_after["digital_gender_gap_analysis"] = np.where(
        panel_after["internet_male_use_analysis"] > 0,
        1 - panel_after["internet_female_use_analysis"] / panel_after["internet_male_use_analysis"],
        np.nan
    )



vars_A_imp = [
    "internet_use_analysis",
    "female_lfp_analysis",
    "female_vulnerable_emp_analysis",
    "gdp_pc_ppp_analysis",
    "fertility",
    "urbanization",
    "service_share"
]

vars_B_imp = vars_A_imp + [
    "trade_open_analysis",
    "wbl_index_analysis"
]

vars_C_imp = vars_B_imp + [
    "female_tertiary_analysis",
    "internet_female_use_analysis",
    "internet_male_use_analysis",
    "digital_gender_gap_analysis"
]

vars_A_imp = [v for v in vars_A_imp if v in panel_after.columns]
vars_B_imp = [v for v in vars_B_imp if v in panel_after.columns]
vars_C_imp = [v for v in vars_C_imp if v in panel_after.columns]

panel_after["sample_A_complete_imp"] = panel_after[vars_A_imp].notna().all(axis=1).astype(int)
panel_after["sample_B_complete_imp"] = panel_after[vars_B_imp].notna().all(axis=1).astype(int)
panel_after["sample_C_complete_imp"] = panel_after[vars_C_imp].notna().all(axis=1).astype(int)

print("\n================ 插补后样本标签汇总 ================\n")
print(panel_after[[
    "sample_A_complete",
    "sample_B_complete",
    "sample_C_complete",
    "sample_A_complete_imp",
    "sample_B_complete_imp",
    "sample_C_complete_imp"
]].sum())


check_cols = [
    "country_name", "iso3", "year",
    "female_tertiary",
    "female_tertiary_imp",
    "female_tertiary_analysis",
    "female_tertiary_imputed_flag"
]
check_cols = [c for c in check_cols if c in panel_after.columns]

print("\n================ Afghanistan 插补检查 ================\n")
print(panel_after.loc[panel_after["iso3"] == "AFG", check_cols].to_string(index=False))

print("\n================ panel_after 概览 ================\n")
print("panel_after 维度：", panel_after.shape)
print("panel_after 列数：", len(panel_after.columns))

safe_to_csv(panel_after, PANEL_AFTER_PATH, "panel_after")

