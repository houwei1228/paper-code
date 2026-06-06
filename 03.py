import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf


DATA_PATH = r"D:/课程/论文/投稿论文/数据/panel_before_imputation_2006_2023.csv"

CORR_PATH = r"D:/课程/论文/投稿论文/数据/correlation_matrix_main_outcome_secure_employment.csv"
VIF_PATH = r"D:/课程/论文/投稿论文/数据/vif_main_outcome_secure_employment.csv"

TWFE_MAIN_COEF_PATH = r"D:/课程/论文/投稿论文/数据/twfe_main_secure_employment_coef_table.csv"
TWFE_MAIN_FOCUS_PATH = r"D:/课程/论文/投稿论文/数据/twfe_main_secure_employment_focus_table.csv"

TWFE_VULN_COEF_PATH = r"D:/课程/论文/投稿论文/数据/twfe_main_female_vulnerable_emp_coef_table.csv"
TWFE_VULN_FOCUS_PATH = r"D:/课程/论文/投稿论文/数据/twfe_main_female_vulnerable_emp_focus_table.csv"

TWFE_EXT_RESULTS_PATH = r"D:/课程/论文/投稿论文/数据/twfe_extended_outcomes_results.csv"



OUTPUT_DIR = r"D:/课程/论文/投稿论文/数据"
os.makedirs(OUTPUT_DIR, exist_ok=True)


PROGRESSION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_progressive_main_table.csv")

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

print("\n================ 原始读取信息 ================\n")
print("数据维度：", df.shape)
print("列数：", len(df.columns))


# 主样本：sample_A_complete == 1

if "sample_A_complete" not in df.columns:
    raise ValueError("数据中不存在 sample_A_complete，请检查 panel_before 文件。")

df_main = df.loc[df["sample_A_complete"] == 1].copy()

print("\n================ 主样本信息（sample_A_complete == 1） ================\n")
print("主样本维度：", df_main.shape)
print("国家数：", df_main["iso3"].nunique())
print("年份数：", df_main["year"].nunique())


main_y_var = "secure_employment"
vuln_y_var = "female_vulnerable_emp"
x_var = "internet_use"
control_vars = ["log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]

common_id_vars = ["country_name", "iso3", "year"]

focus_vars = [x_var] + control_vars


#主结果变量：secure_employment

main_model_vars = [main_y_var, x_var] + control_vars + common_id_vars
missing_vars = [v for v in main_model_vars if v not in df_main.columns]
if missing_vars:
    raise ValueError(f"以下变量在数据中不存在：{missing_vars}")

df_main_model = df_main[main_model_vars].dropna().copy()

print("\n================ 主结果变量回归样本信息 ================\n")
print("结果变量：", main_y_var)
print("回归样本维度：", df_main_model.shape)
print("国家数：", df_main_model["iso3"].nunique())
print("年份数：", df_main_model["year"].nunique())


#  相关系数矩阵（基于 secure_employment）

corr_vars = [main_y_var, x_var] + control_vars
corr_matrix = df_main_model[corr_vars].corr()

print("\n================ 相关系数矩阵（主结果变量：secure_employment） ================\n")
print(corr_matrix.round(4).to_string())

corr_matrix.to_csv(CORR_PATH, encoding="utf-8-sig")


#  VIF（pooled 诊断，不含FE）

X_vif = df_main_model[[x_var] + control_vars].copy()
X_vif_const = sm.add_constant(X_vif)

vif_df = pd.DataFrame()
vif_df["variable"] = X_vif_const.columns
vif_df["VIF"] = [
    variance_inflation_factor(X_vif_const.values, i)
    for i in range(X_vif_const.shape[1])
]

print("\n================ VIF 结果（主结果变量模型，pooled，不含FE） ================\n")
print(vif_df.round(4).to_string(index=False))

vif_df.to_csv(VIF_PATH, index=False, encoding="utf-8-sig")


# TWFE 

def run_twfe(df_input, y_var, x_var, control_vars):
    use_vars = [y_var, x_var] + control_vars + ["iso3", "year"]
    missing_vars = [v for v in use_vars if v not in df_input.columns]
    if missing_vars:
        raise ValueError(f"{y_var} 回归缺少变量：{missing_vars}")

    df_model = df_input[use_vars].dropna().copy()

    formula = (
        f"{y_var} ~ {x_var} + "
        + " + ".join(control_vars)
        + " + C(iso3) + C(year)"
    )

    model = sm.OLS.from_formula(formula=formula, data=df_model)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_model["iso3"]}
    )

    coef_table = pd.DataFrame({
        "variable": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "t_value": result.tvalues.values,
        "p_value": result.pvalues.values
    })

    focus_table = coef_table.loc[coef_table["variable"].isin([x_var] + control_vars)].copy()

    return {
        "df_model": df_model,
        "formula": formula,
        "result": result,
        "coef_table": coef_table,
        "focus_table": focus_table
    }


#  secure_employment TWFE

main_out = run_twfe(df_main, main_y_var, x_var, control_vars)

print("\n================ 主结果变量 TWFE 回归公式 ================\n")
print(main_out["formula"])

print("\n================ 主结果变量基准 TWFE 结果（按国家聚类标准误） ================\n")
print(main_out["result"].summary())

print("\n================ 主结果变量系数表（前20行） ================\n")
print(main_out["coef_table"].head(20).round(4).to_string(index=False))

main_out["coef_table"].to_csv(TWFE_MAIN_COEF_PATH, index=False, encoding="utf-8-sig")

print("\n================ 主结果变量关注变量结果 ================\n")
print(main_out["focus_table"].round(4).to_string(index=False))

main_out["focus_table"].to_csv(TWFE_MAIN_FOCUS_PATH, index=False, encoding="utf-8-sig")

print("\n================ 主结果变量样本统计 ================\n")
print(f"N = {int(main_out['result'].nobs)}")
print(f"Countries = {main_out['df_model']['iso3'].nunique()}")
print(f"Years = {main_out['df_model']['year'].nunique()}")
print(f"R-squared = {main_out['result'].rsquared:.4f}")
print(f"Adj. R-squared = {main_out['result'].rsquared_adj:.4f}")


# female_vulnerable_emp

print("\n" + "=" * 70)
print("重点结果变量：female_vulnerable_emp")
print("=" * 70)

if vuln_y_var not in df_main.columns:
    raise ValueError(f"数据中不存在 {vuln_y_var}")

vuln_out = run_twfe(df_main, vuln_y_var, x_var, control_vars)

print("\n================ female_vulnerable_emp TWFE 回归公式 ================\n")
print(vuln_out["formula"])

print("\n================ female_vulnerable_emp TWFE 结果（按国家聚类标准误） ================\n")
print(vuln_out["result"].summary())

print("\n================ female_vulnerable_emp 系数表（前20行） ================\n")
print(vuln_out["coef_table"].head(20).round(4).to_string(index=False))

vuln_out["coef_table"].to_csv(TWFE_VULN_COEF_PATH, index=False, encoding="utf-8-sig")

print("\n================ female_vulnerable_emp 关注变量结果 ================\n")
print(vuln_out["focus_table"].round(4).to_string(index=False))

vuln_out["focus_table"].to_csv(TWFE_VULN_FOCUS_PATH, index=False, encoding="utf-8-sig")

print("\n================ female_vulnerable_emp 样本统计 ================\n")
print(f"N = {int(vuln_out['result'].nobs)}")
print(f"Countries = {vuln_out['df_model']['iso3'].nunique()}")
print(f"Years = {vuln_out['df_model']['year'].nunique()}")
print(f"R-squared = {vuln_out['result'].rsquared:.4f}")
print(f"Adj. R-squared = {vuln_out['result'].rsquared_adj:.4f}")

# 扩展结果变量
extended_outcomes = [
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]

all_results = []

for _, row in main_out["focus_table"].iterrows():
    all_results.append({
        "outcome_type": "main",
        "outcome": main_y_var,
        "variable": row["variable"],
        "coef": row["coef"],
        "std_err": row["std_err"],
        "t_value": row["t_value"],
        "p_value": row["p_value"],
        "n_obs": int(main_out["result"].nobs),
        "n_country": main_out["df_model"]["iso3"].nunique(),
        "n_year": main_out["df_model"]["year"].nunique(),
        "r_squared": main_out["result"].rsquared
    })


for _, row in vuln_out["focus_table"].iterrows():
    all_results.append({
        "outcome_type": "priority_main",
        "outcome": vuln_y_var,
        "variable": row["variable"],
        "coef": row["coef"],
        "std_err": row["std_err"],
        "t_value": row["t_value"],
        "p_value": row["p_value"],
        "n_obs": int(vuln_out["result"].nobs),
        "n_country": vuln_out["df_model"]["iso3"].nunique(),
        "n_year": vuln_out["df_model"]["year"].nunique(),
        "r_squared": vuln_out["result"].rsquared
    })


for y_var in extended_outcomes:
    if y_var not in df_main.columns:
        print(f"\n跳过 {y_var}：数据中不存在该变量。")
        continue

    print("\n" + "=" * 70)
    print(f"扩展结果变量：{y_var}")
    print("=" * 70)

    out = run_twfe(df_main, y_var, x_var, control_vars)

    print("样本量：", out["df_model"].shape[0])
    print("国家数：", out["df_model"]["iso3"].nunique())
    print("年份数：", out["df_model"]["year"].nunique())

    print("\n回归公式：")
    print(out["formula"])

    print("\n================ 扩展结果变量 TWFE 结果 ================\n")
    print(out["result"].summary())

    print("\n================ 扩展结果变量关注变量结果 ================\n")
    print(out["focus_table"].round(4).to_string(index=False))

    for _, row in out["focus_table"].iterrows():
        all_results.append({
            "outcome_type": "extended",
            "outcome": y_var,
            "variable": row["variable"],
            "coef": row["coef"],
            "std_err": row["std_err"],
            "t_value": row["t_value"],
            "p_value": row["p_value"],
            "n_obs": int(out["result"].nobs),
            "n_country": out["df_model"]["iso3"].nunique(),
            "n_year": out["df_model"]["year"].nunique(),
            "r_squared": out["result"].rsquared
        })

# 汇总结果表

results_df = pd.DataFrame(all_results)

print("\n================ secure_employment 与 female_vulnerable_emp 对照 ================\n")
compare_df = results_df[
    results_df["outcome"].isin([main_y_var, vuln_y_var]) &
    (results_df["variable"] == x_var)
].copy()

if not compare_df.empty:
    print(compare_df.round(4).to_string(index=False))
else:
    print("没有成功生成 secure_employment / female_vulnerable_emp 的对照结果。")

print("\n================ 主结果 + 优先结果 + 扩展结果 核心结果汇总 ================\n")
if not results_df.empty:
    print(results_df.round(4).to_string(index=False))
else:
    print("没有成功生成结果。")

results_df.to_csv(TWFE_EXT_RESULTS_PATH, index=False, encoding="utf-8-sig")
print("\n结果已保存：", TWFE_EXT_RESULTS_PATH)


# 加入一期滞后 internet_use：L.internet_use

print("\n" + "=" * 70)
print("滞后一期核心解释变量：L.internet_use")
print("=" * 70)


df_lag = df_main.copy()
df_lag = df_lag.sort_values(["iso3", "year"]).reset_index(drop=True)


df_lag["internet_use_l1"] = df_lag.groupby("iso3")["internet_use"].shift(1)


lag_outcomes = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]

lag_outcomes = [v for v in lag_outcomes if v in df_lag.columns]

lag_results_list = []

for outcome in lag_outcomes:
    print("\n" + "=" * 70)
    print(f"滞后解释变量回归：{outcome}")
    print("=" * 70)

    needed_cols = [
        outcome,
        "internet_use_l1",
        "log_gdp_pc_ppp",
        "fertility",
        "urbanization",
        "service_share",
        "iso3",
        "year"
    ]

    reg_df = df_lag[needed_cols].dropna().copy()

    print("样本量：", reg_df.shape[0])
    print("国家数：", reg_df["iso3"].nunique())
    print("年份数：", reg_df["year"].nunique())

    formula_lag = (
        f"{outcome} ~ internet_use_l1 + log_gdp_pc_ppp + fertility + "
        f"urbanization + service_share + C(iso3) + C(year)"
    )

    print("\n回归公式：")
    print(formula_lag)

    model_lag = smf.ols(formula_lag, data=reg_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["iso3"]}
    )

    print("\n================ 滞后一期 TWFE 结果 ================\n")
    print(model_lag.summary())

    coef_table_lag = pd.DataFrame({
        "variable": model_lag.params.index,
        "coef": model_lag.params.values,
        "std_err": model_lag.bse.values,
        "t_value": model_lag.tvalues.values,
        "p_value": model_lag.pvalues.values
    })

    key_vars = ["internet_use_l1", "log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
    coef_focus_lag = coef_table_lag[coef_table_lag["variable"].isin(key_vars)].copy()

    print("\n================ 滞后一期关注变量结果 ================\n")
    print(coef_focus_lag.to_string(index=False))

    for _, row in coef_focus_lag.iterrows():
        lag_results_list.append({
            "outcome": outcome,
            "variable": row["variable"],
            "coef": row["coef"],
            "std_err": row["std_err"],
            "t_value": row["t_value"],
            "p_value": row["p_value"],
            "n_obs": reg_df.shape[0],
            "n_country": reg_df["iso3"].nunique(),
            "n_year": reg_df["year"].nunique(),
            "r_squared": model_lag.rsquared
        })

lag_results_df = pd.DataFrame(lag_results_list)

print("\n================ 滞后一期结果汇总 ================\n")
print(lag_results_df.to_string(index=False))

LAG_RESULTS_PATH = r"D:/课程/论文/投稿论文/数据/twfe_lag1_internet_results.csv"
lag_results_df.to_csv(LAG_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"滞后一期TWFE结果已保存：{LAG_RESULTS_PATH}")


#按公式回归并提取核心变量结果

def run_formula_cluster(reg_df, formula, key_vars, cluster_var="iso3"):
    model = smf.ols(formula=formula, data=reg_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df[cluster_var]}
    )

    coef_table = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values
    })

    focus_table = coef_table.loc[coef_table["variable"].isin(key_vars)].copy()

    return model, coef_table, focus_table

#  Country-specific linear trends

TREND_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_country_specific_trends_results.csv")

print("\n" + "=" * 70)
print("稳健性检验：Country-specific linear trends")
print("=" * 70)

df_trend = df_main.copy()
df_trend = df_trend.sort_values(["iso3", "year"]).reset_index(drop=True)


df_trend["year_trend"] = df_trend["year"] - df_trend["year"].min()

trend_outcomes = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]
trend_outcomes = [v for v in trend_outcomes if v in df_trend.columns]

trend_results_list = []

for outcome in trend_outcomes:
    print("\n" + "-" * 60)
    print(f"Country trends 回归：{outcome}")

    needed_cols = [
        outcome, "internet_use",
        "log_gdp_pc_ppp", "fertility", "urbanization", "service_share",
        "iso3", "year", "year_trend"
    ]
    reg_df = df_trend[needed_cols].dropna().copy()

    formula_trend = (
        f"{outcome} ~ internet_use + log_gdp_pc_ppp + fertility + "
        f"urbanization + service_share + C(iso3) + C(year) + C(iso3):year_trend"
    )

    print("样本量：", reg_df.shape[0])
    print("国家数：", reg_df["iso3"].nunique())
    print("年份数：", reg_df["year"].nunique())
    print("公式：", formula_trend)

    model_trend, coef_table_trend, focus_table_trend = run_formula_cluster(
        reg_df=reg_df,
        formula=formula_trend,
        key_vars=["internet_use", "log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
    )

    print("\n关注变量结果：")
    print(focus_table_trend.round(4).to_string(index=False))

    for _, row in focus_table_trend.iterrows():
        trend_results_list.append({
            "spec": "country_specific_linear_trends",
            "outcome": outcome,
            "variable": row["variable"],
            "coef": row["coef"],
            "std_err": row["std_err"],
            "t_value": row["t_value"],
            "p_value": row["p_value"],
            "n_obs": int(model_trend.nobs),
            "n_country": reg_df["iso3"].nunique(),
            "n_year": reg_df["year"].nunique(),
            "r_squared": model_trend.rsquared
        })

trend_results_df = pd.DataFrame(trend_results_list)
trend_results_df.to_csv(TREND_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"\nCountry-specific linear trends 结果已保存：{TREND_RESULTS_PATH}")


# First-difference / Long-difference

DIFF_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_difference_results.csv")

print("\n" + "=" * 70)
print("稳健性检验：First-difference / Long-difference")
print("=" * 70)

def build_diff_panel(data, vars_to_diff, id_var="iso3", time_var="year", diff_lag=1):
    df_tmp = data.sort_values([id_var, time_var]).copy()
    for v in vars_to_diff:
        df_tmp[f"D{diff_lag}_{v}"] = df_tmp.groupby(id_var)[v].diff(diff_lag)
    return df_tmp

diff_outcomes = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]
diff_outcomes = [v for v in diff_outcomes if v in df_main.columns]

diff_results_list = []


diff_lags = [1, 3]

for diff_lag in diff_lags:
    print("\n" + "-" * 60)
    print(f"开始差分规格：lag = {diff_lag}")

    vars_to_diff = diff_outcomes + ["internet_use", "log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
    df_diff = build_diff_panel(df_main.copy(), vars_to_diff=vars_to_diff, diff_lag=diff_lag)

    for outcome in diff_outcomes:
        y_diff = f"D{diff_lag}_{outcome}"
        x_diff = f"D{diff_lag}_internet_use"
        control_diffs = [
            f"D{diff_lag}_log_gdp_pc_ppp",
            f"D{diff_lag}_fertility",
            f"D{diff_lag}_urbanization",
            f"D{diff_lag}_service_share"
        ]

        needed_cols = [y_diff, x_diff] + control_diffs + ["iso3", "year"]
        reg_df = df_diff[needed_cols].dropna().copy()

        if reg_df.empty:
            continue

       
        formula_diff = (
            f"{y_diff} ~ {x_diff} + "
            + " + ".join(control_diffs)
            + " + C(year)"
        )

        print("\n" + "-" * 40)
        print(f"因变量：{outcome} | 差分长度：{diff_lag}")
        print("样本量：", reg_df.shape[0])
        print("公式：", formula_diff)

        model_diff, coef_table_diff, focus_table_diff = run_formula_cluster(
            reg_df=reg_df,
            formula=formula_diff,
            key_vars=[x_diff] + control_diffs
        )

        print("关注变量结果：")
        print(focus_table_diff.round(4).to_string(index=False))

        for _, row in focus_table_diff.iterrows():
            diff_results_list.append({
                "spec": f"{diff_lag}_difference",
                "outcome": outcome,
                "variable": row["variable"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "t_value": row["t_value"],
                "p_value": row["p_value"],
                "n_obs": int(model_diff.nobs),
                "n_country": reg_df["iso3"].nunique(),
                "n_year": reg_df["year"].nunique(),
                "r_squared": model_diff.rsquared
            })

diff_results_df = pd.DataFrame(diff_results_list)
diff_results_df.to_csv(DIFF_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"\nDifference specifications 结果已保存：{DIFF_RESULTS_PATH}")


#  Dynamic lead-lag checks

DYNAMIC_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_dynamic_lead_lag_results.csv")

print("\n" + "=" * 70)
print("稳健性检验：Dynamic lead-lag checks")
print("=" * 70)

df_dyn = df_main.copy().sort_values(["iso3", "year"]).reset_index(drop=True)


df_dyn["internet_use_f1"] = df_dyn.groupby("iso3")["internet_use"].shift(-1)  # lead 1
df_dyn["internet_use_l1"] = df_dyn.groupby("iso3")["internet_use"].shift(1)
df_dyn["internet_use_l2"] = df_dyn.groupby("iso3")["internet_use"].shift(2)

dyn_outcomes = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]
dyn_outcomes = [v for v in dyn_outcomes if v in df_dyn.columns]

dyn_results_list = []

for outcome in dyn_outcomes:
    needed_cols = [
        outcome, "internet_use_f1", "internet_use", "internet_use_l1", "internet_use_l2",
        "log_gdp_pc_ppp", "fertility", "urbanization", "service_share",
        "iso3", "year"
    ]
    reg_df = df_dyn[needed_cols].dropna().copy()

    if reg_df.empty:
        continue

    formula_dyn = (
        f"{outcome} ~ internet_use_f1 + internet_use + internet_use_l1 + internet_use_l2 + "
        f"log_gdp_pc_ppp + fertility + urbanization + service_share + C(iso3) + C(year)"
    )

    print("\n" + "-" * 40)
    print(f"动态规格：{outcome}")
    print("样本量：", reg_df.shape[0])
    print("公式：", formula_dyn)

    model_dyn, coef_table_dyn, focus_table_dyn = run_formula_cluster(
        reg_df=reg_df,
        formula=formula_dyn,
        key_vars=["internet_use_f1", "internet_use", "internet_use_l1", "internet_use_l2"]
    )

    print("关注变量结果：")
    print(focus_table_dyn.round(4).to_string(index=False))

    for _, row in focus_table_dyn.iterrows():
        dyn_results_list.append({
            "spec": "dynamic_lead_lag",
            "outcome": outcome,
            "variable": row["variable"],
            "coef": row["coef"],
            "std_err": row["std_err"],
            "t_value": row["t_value"],
            "p_value": row["p_value"],
            "n_obs": int(model_dyn.nobs),
            "n_country": reg_df["iso3"].nunique(),
            "n_year": reg_df["year"].nunique(),
            "r_squared": model_dyn.rsquared
        })

dyn_results_df = pd.DataFrame(dyn_results_list)
dyn_results_df.to_csv(DYNAMIC_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"\nDynamic lead-lag 结果已保存：{DYNAMIC_RESULTS_PATH}")

# Period-split checks

PERIOD_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_period_split_results.csv")

print("\n" + "=" * 70)
print("稳健性检验：Period-split checks")
print("=" * 70)

period_defs = {
    "2006_2014": (2006, 2014),
    "2015_2023": (2015, 2023)
}

period_outcomes = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]
period_outcomes = [v for v in period_outcomes if v in df_main.columns]

period_results_list = []

for period_name, (start_y, end_y) in period_defs.items():
    df_period = df_main[(df_main["year"] >= start_y) & (df_main["year"] <= end_y)].copy()

    print("\n" + "-" * 60)
    print(f"样本期间：{period_name}")

    for outcome in period_outcomes:
        needed_cols = [
            outcome, "internet_use",
            "log_gdp_pc_ppp", "fertility", "urbanization", "service_share",
            "iso3", "year"
        ]
        reg_df = df_period[needed_cols].dropna().copy()

        if reg_df.empty:
            continue

        formula_period = (
            f"{outcome} ~ internet_use + log_gdp_pc_ppp + fertility + "
            f"urbanization + service_share + C(iso3) + C(year)"
        )

        model_period, coef_table_period, focus_table_period = run_formula_cluster(
            reg_df=reg_df,
            formula=formula_period,
            key_vars=["internet_use", "log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
        )

        print("\n" + "-" * 40)
        print(f"因变量：{outcome} | 分期：{period_name}")
        print("样本量：", reg_df.shape[0])
        print(focus_table_period.round(4).to_string(index=False))

        for _, row in focus_table_period.iterrows():
            period_results_list.append({
                "spec": "period_split",
                "period": period_name,
                "outcome": outcome,
                "variable": row["variable"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "t_value": row["t_value"],
                "p_value": row["p_value"],
                "n_obs": int(model_period.nobs),
                "n_country": reg_df["iso3"].nunique(),
                "n_year": reg_df["year"].nunique(),
                "r_squared": model_period.rsquared
            })

period_results_df = pd.DataFrame(period_results_list)
period_results_df.to_csv(PERIOD_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"\nPeriod-split 结果已保存：{PERIOD_RESULTS_PATH}")



# secure_employment & female_vulnerable_emp


import statsmodels.formula.api as smf

print("\n" + "=" * 70)
print("递进式规格表：secure_employment 与 female_vulnerable_emp")
print("=" * 70)


PROGRESSION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_progressive_main_table.csv")

def safe_to_csv(df, path, desc="文件"):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{desc}已保存：{path}")


df_prog = df[df["sample_A_complete"] == 1].copy()

main_x = "internet_use"
controls = ["log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
main_outcomes = ["secure_employment", "female_vulnerable_emp"]

specs = [
    {
        "spec_id": "col1",
        "spec_name": "Internet only",
        "rhs": f"{main_x}",
        "has_controls": 0,
        "has_country_fe": 0,
        "has_year_fe": 0,
    },
    {
        "spec_id": "col2",
        "spec_name": "Internet + Controls",
        "rhs": f"{main_x} + " + " + ".join(controls),
        "has_controls": 1,
        "has_country_fe": 0,
        "has_year_fe": 0,
    },
    {
        "spec_id": "col3",
        "spec_name": "Internet + FE",
        "rhs": f"{main_x} + C(iso3) + C(year)",
        "has_controls": 0,
        "has_country_fe": 1,
        "has_year_fe": 1,
    },
    {
        "spec_id": "col4",
        "spec_name": "Internet + Controls + FE",
        "rhs": f"{main_x} + " + " + ".join(controls) + " + C(iso3) + C(year)",
        "has_controls": 1,
        "has_country_fe": 1,
        "has_year_fe": 1,
    },
]


#  回归函数

def run_progressive_spec(data, y, spec):
    needed_cols = [y, main_x, "iso3", "year"] + controls
    reg_df = data[needed_cols].dropna().copy()

    formula = f"{y} ~ {spec['rhs']}"
    print("\n" + "-" * 60)
    print(f"因变量：{y} | 规格：{spec['spec_id']} - {spec['spec_name']}")
    print(f"样本量：{len(reg_df)}")
    print("回归公式：")
    print(formula)

    model = smf.ols(formula=formula, data=reg_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["iso3"]}
    )

    coef = model.params.get(main_x, float("nan"))
    se = model.bse.get(main_x, float("nan"))
    tval = model.tvalues.get(main_x, float("nan"))
    pval = model.pvalues.get(main_x, float("nan"))

    row = {
        "outcome": y,
        "spec_id": spec["spec_id"],
        "spec_name": spec["spec_name"],
        "coef_internet_use": coef,
        "std_err": se,
        "t_value": tval,
        "p_value": pval,
        "n_obs": int(model.nobs),
        "n_country": reg_df["iso3"].nunique(),
        "n_year": reg_df["year"].nunique(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "has_controls": spec["has_controls"],
        "country_fe": spec["has_country_fe"],
        "year_fe": spec["has_year_fe"],
        "formula": formula
    }
    return row, model


# 四列递进式规格

progressive_rows = []

for y in main_outcomes:
    print("\n" + "=" * 70)
    print(f"开始递进式规格：{y}")
    print("=" * 70)

    for spec in specs:
        row, model = run_progressive_spec(df_prog, y, spec)
        progressive_rows.append(row)

        print("\n核心解释变量结果：")
        print(pd.DataFrame([{
            "variable": main_x,
            "coef": row["coef_internet_use"],
            "std_err": row["std_err"],
            "t_value": row["t_value"],
            "p_value": row["p_value"],
            "n_obs": row["n_obs"],
            "r_squared": row["r_squared"]
        }]).round(4))


progressive_results_df = pd.DataFrame(progressive_rows)

print("\n" + "=" * 70)
print("递进式规格长表")
print("=" * 70)
print(progressive_results_df.round(4))

# 宽表

main_table = progressive_results_df.pivot(index="outcome", columns="spec_id", values="coef_internet_use")
main_table = main_table.rename(columns={
    "col1": "col1_coef",
    "col2": "col2_coef",
    "col3": "col3_coef",
    "col4": "col4_coef"
})

main_table_se = progressive_results_df.pivot(index="outcome", columns="spec_id", values="std_err")
main_table_se = main_table_se.rename(columns={
    "col1": "col1_se",
    "col2": "col2_se",
    "col3": "col3_se",
    "col4": "col4_se"
})

main_table_p = progressive_results_df.pivot(index="outcome", columns="spec_id", values="p_value")
main_table_p = main_table_p.rename(columns={
    "col1": "col1_p",
    "col2": "col2_p",
    "col3": "col3_p",
    "col4": "col4_p"
})

main_table_n = progressive_results_df.pivot(index="outcome", columns="spec_id", values="n_obs")
main_table_n = main_table_n.rename(columns={
    "col1": "col1_n",
    "col2": "col2_n",
    "col3": "col3_n",
    "col4": "col4_n"
})

main_table_r2 = progressive_results_df.pivot(index="outcome", columns="spec_id", values="r_squared")
main_table_r2 = main_table_r2.rename(columns={
    "col1": "col1_r2",
    "col2": "col2_r2",
    "col3": "col3_r2",
    "col4": "col4_r2"
})

main_table_final = (
    main_table
    .join(main_table_se)
    .join(main_table_p)
    .join(main_table_n)
    .join(main_table_r2)
    .reset_index()
)


# 增加显著性星号

def add_stars(coef, p):
    if pd.isna(coef) or pd.isna(p):
        return ""
    if p < 0.01:
        return f"{coef:.4f}***"
    elif p < 0.05:
        return f"{coef:.4f}**"
    elif p < 0.10:
        return f"{coef:.4f}*"
    else:
        return f"{coef:.4f}"

for c in ["col1", "col2", "col3", "col4"]:
    main_table_final[f"{c}_coef_star"] = main_table_final.apply(
        lambda x: add_stars(x[f"{c}_coef"], x[f"{c}_p"]), axis=1
    )
    main_table_final[f"{c}_se_fmt"] = main_table_final[f"{c}_se"].apply(
        lambda v: f"({v:.4f})" if pd.notna(v) else ""
    )

print("\n" + "=" * 70)
print("适合主表整理的宽表")
print("=" * 70)
print(main_table_final.round(4))

# 最终主表

presentation_rows = []

for _, row in main_table_final.iterrows():
    y = row["outcome"]

    coef_row = {
        "outcome": y,
        "stat": "internet_use"
    }
    se_row = {
        "outcome": y,
        "stat": "std_err"
    }
    n_row = {
        "outcome": y,
        "stat": "N"
    }
    r2_row = {
        "outcome": y,
        "stat": "R2"
    }
    ctrl_row = {
        "outcome": y,
        "stat": "Controls"
    }
    cfe_row = {
        "outcome": y,
        "stat": "Country FE"
    }
    yfe_row = {
        "outcome": y,
        "stat": "Year FE"
    }

    for i, spec in enumerate(specs, start=1):
        sid = spec["spec_id"]
        colname = f"col{i}"

        coef_row[colname] = row[f"{sid}_coef_star"]
        se_row[colname] = row[f"{sid}_se_fmt"]
        n_row[colname] = int(row[f"{sid}_n"]) if pd.notna(row[f"{sid}_n"]) else ""
        r2_row[colname] = f"{row[f'{sid}_r2']:.4f}" if pd.notna(row[f"{sid}_r2"]) else ""
        ctrl_row[colname] = "Yes" if spec["has_controls"] == 1 else "No"
        cfe_row[colname] = "Yes" if spec["has_country_fe"] == 1 else "No"
        yfe_row[colname] = "Yes" if spec["has_year_fe"] == 1 else "No"

    presentation_rows.extend([coef_row, se_row, n_row, r2_row, ctrl_row, cfe_row, yfe_row])

presentation_table_df = pd.DataFrame(presentation_rows)

print("\n" + "=" * 70)
print("论文主表展示版")
print("=" * 70)
print(presentation_table_df)


safe_to_csv(progressive_results_df, PROGRESSION_RESULTS_PATH.replace(".csv", "_long.csv"), "递进式规格长表")
safe_to_csv(main_table_final, PROGRESSION_RESULTS_PATH.replace(".csv", "_wide.csv"), "递进式规格宽表")
safe_to_csv(presentation_table_df, PROGRESSION_RESULTS_PATH, "递进式规格主表展示版")


#vulnerable / secure 双结果主表

DUAL_MAIN_TABLE_PATH = os.path.join(OUTPUT_DIR, "twfe_dual_outcome_main_table.csv")

print("\n" + "=" * 70)
print("vulnerable / secure 双结果主表")
print("=" * 70)


dual_outcomes_order = ["female_vulnerable_emp", "secure_employment"]
dual_outcome_labels = {
    "female_vulnerable_emp": "Female vulnerable employment",
    "secure_employment": "Secure employment"
}

dual_df = progressive_results_df[
    progressive_results_df["outcome"].isin(dual_outcomes_order)
].copy()

dual_df["outcome"] = pd.Categorical(
    dual_df["outcome"],
    categories=dual_outcomes_order,
    ordered=True
)

dual_df["spec_id"] = pd.Categorical(
    dual_df["spec_id"],
    categories=["col1", "col2", "col3", "col4"],
    ordered=True
)

dual_df = dual_df.sort_values(["outcome", "spec_id"]).copy()


def add_stars(coef, p):
    if pd.isna(coef) or pd.isna(p):
        return ""
    if p < 0.01:
        return f"{coef:.4f}***"
    elif p < 0.05:
        return f"{coef:.4f}**"
    elif p < 0.10:
        return f"{coef:.4f}*"
    else:
        return f"{coef:.4f}"

dual_rows = []

for outcome in dual_outcomes_order:
    sub = dual_df[dual_df["outcome"] == outcome].sort_values("spec_id").copy()

    coef_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "internet_use"
    }
    se_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "std_err"
    }
    n_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "N"
    }
    country_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "Countries"
    }
    year_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "Years"
    }
    r2_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "R2"
    }
    ctrl_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "Controls"
    }
    cfe_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "Country FE"
    }
    yfe_row = {
        "outcome_block": dual_outcome_labels[outcome],
        "stat": "Year FE"
    }

    for sid in ["col1", "col2", "col3", "col4"]:
        cell = sub[sub["spec_id"] == sid]

        if len(cell) == 0:
            coef_row[sid] = ""
            se_row[sid] = ""
            n_row[sid] = ""
            country_row[sid] = ""
            year_row[sid] = ""
            r2_row[sid] = ""
            ctrl_row[sid] = ""
            cfe_row[sid] = ""
            yfe_row[sid] = ""
            continue

        cell = cell.iloc[0]

        coef_row[sid] = add_stars(cell["coef_internet_use"], cell["p_value"])
        se_row[sid] = f"({cell['std_err']:.4f})" if pd.notna(cell["std_err"]) else ""
        n_row[sid] = int(cell["n_obs"]) if pd.notna(cell["n_obs"]) else ""
        country_row[sid] = int(cell["n_country"]) if pd.notna(cell["n_country"]) else ""
        year_row[sid] = int(cell["n_year"]) if pd.notna(cell["n_year"]) else ""
        r2_row[sid] = f"{cell['r_squared']:.4f}" if pd.notna(cell["r_squared"]) else ""
        ctrl_row[sid] = "Yes" if int(cell["has_controls"]) == 1 else "No"
        cfe_row[sid] = "Yes" if int(cell["country_fe"]) == 1 else "No"
        yfe_row[sid] = "Yes" if int(cell["year_fe"]) == 1 else "No"

    dual_rows.extend([
        coef_row,
        se_row,
        n_row,
        country_row,
        year_row,
        r2_row,
        ctrl_row,
        cfe_row,
        yfe_row,
        {"outcome_block": "", "stat": "", "col1": "", "col2": "", "col3": "", "col4": ""}
    ])

dual_main_table_df = pd.DataFrame(dual_rows)

print("\n" + "=" * 70)
print("双结果主表（适合复制到论文）")
print("=" * 70)
print(dual_main_table_df)

safe_to_csv(dual_main_table_df, DUAL_MAIN_TABLE_PATH, "vulnerable_secure双结果主表")


# 替换核心解释变量的稳健性检验
# 核心替代变量：fixed_broadband / mobile_subscriptions
# 主因变量：secure_employment / female_vulnerable_emp

ALT_X_RESULTS_PATH = os.path.join(OUTPUT_DIR, "twfe_alternative_core_x_results.csv")
ALT_X_MAIN_TABLE_PATH = os.path.join(OUTPUT_DIR, "twfe_alternative_core_x_main_table.csv")

print("\n" + "=" * 70)
print("替换核心解释变量的稳健性检验")
print("=" * 70)



df_altx = df[df["sample_A_complete"] == 1].copy()
alt_outcomes = ["female_vulnerable_emp", "secure_employment"]
candidate_alt_x = [
    "fixed_broadband",
    "mobile_subscriptions"
]
controls = ["log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]


available_alt_x = [x for x in candidate_alt_x if x in df_altx.columns]
missing_alt_x = [x for x in candidate_alt_x if x not in df_altx.columns]

print("\n可用于替换的核心解释变量：", available_alt_x)
if len(missing_alt_x) > 0:
    print("以下变量不存在，已自动跳过：", missing_alt_x)

if len(available_alt_x) == 0:
    print("\n没有找到可用的替代核心解释变量，代码停止。")
else:

    def run_alt_core_x_reg(data, y, xvar):
        needed_cols = [y, xvar, "iso3", "year"] + controls
        reg_df = data[needed_cols].dropna().copy()

        formula = f"{y} ~ {xvar} + " + " + ".join(controls) + " + C(iso3) + C(year)"

        print("\n" + "-" * 60)
        print(f"因变量：{y}")
        print(f"替代核心解释变量：{xvar}")
        print(f"样本量：{len(reg_df)}")
        print("回归公式：")
        print(formula)

        model = smf.ols(formula=formula, data=reg_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": reg_df["iso3"]}
        )

        row = {
            "outcome": y,
            "core_x": xvar,
            "coef": model.params.get(xvar, float("nan")),
            "std_err": model.bse.get(xvar, float("nan")),
            "t_value": model.tvalues.get(xvar, float("nan")),
            "p_value": model.pvalues.get(xvar, float("nan")),
            "n_obs": int(model.nobs),
            "n_country": reg_df["iso3"].nunique(),
            "n_year": reg_df["year"].nunique(),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "formula": formula
        }

        return row, model

 
    alt_x_rows = []

    for y in alt_outcomes:
        print("\n" + "=" * 70)
        print(f"替换核心解释变量稳健性：{y}")
        print("=" * 70)

        for xvar in available_alt_x:
            row, model = run_alt_core_x_reg(df_altx, y, xvar)
            alt_x_rows.append(row)

            print("\n关注变量结果：")
            print(pd.DataFrame([{
                "variable": xvar,
                "coef": row["coef"],
                "std_err": row["std_err"],
                "t_value": row["t_value"],
                "p_value": row["p_value"],
                "n_obs": row["n_obs"],
                "r_squared": row["r_squared"]
            }]).round(4))

    alt_x_results_df = pd.DataFrame(alt_x_rows)

    print("\n" + "=" * 70)
    print("替换核心解释变量结果汇总")
    print("=" * 70)
    print(alt_x_results_df.round(4))


    def add_stars(coef, p):
        if pd.isna(coef) or pd.isna(p):
            return ""
        if p < 0.01:
            return f"{coef:.4f}***"
        elif p < 0.05:
            return f"{coef:.4f}**"
        elif p < 0.10:
            return f"{coef:.4f}*"
        else:
            return f"{coef:.4f}"

    alt_x_results_df["coef_star"] = alt_x_results_df.apply(
        lambda r: add_stars(r["coef"], r["p_value"]), axis=1
    )
    alt_x_results_df["se_fmt"] = alt_x_results_df["std_err"].apply(
        lambda v: f"({v:.4f})" if pd.notna(v) else ""
    )
    alt_x_results_df["outcome_label"] = alt_x_results_df["outcome"].map({
        "female_vulnerable_emp": "Female vulnerable employment",
        "secure_employment": "Secure employment"
    })

  
    alt_x_table_rows = []

    outcome_order = ["female_vulnerable_emp", "secure_employment"]

    for outcome in outcome_order:
        sub = alt_x_results_df[alt_x_results_df["outcome"] == outcome].copy()

        coef_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Core explanatory variable"
        }
        se_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Std. Err."
        }
        n_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "N"
        }
        country_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Countries"
        }
        year_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Years"
        }
        r2_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "R2"
        }
        ctrl_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Controls"
        }
        cfe_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Country FE"
        }
        yfe_row = {
            "outcome_block": {
                "female_vulnerable_emp": "Female vulnerable employment",
                "secure_employment": "Secure employment"
            }[outcome],
            "stat": "Year FE"
        }

        for xvar in available_alt_x:
            cell = sub[sub["core_x"] == xvar]

            if len(cell) == 0:
                coef_row[xvar] = ""
                se_row[xvar] = ""
                n_row[xvar] = ""
                country_row[xvar] = ""
                year_row[xvar] = ""
                r2_row[xvar] = ""
                ctrl_row[xvar] = ""
                cfe_row[xvar] = ""
                yfe_row[xvar] = ""
                continue

            cell = cell.iloc[0]
            coef_row[xvar] = cell["coef_star"]
            se_row[xvar] = cell["se_fmt"]
            n_row[xvar] = int(cell["n_obs"]) if pd.notna(cell["n_obs"]) else ""
            country_row[xvar] = int(cell["n_country"]) if pd.notna(cell["n_country"]) else ""
            year_row[xvar] = int(cell["n_year"]) if pd.notna(cell["n_year"]) else ""
            r2_row[xvar] = f"{cell['r_squared']:.4f}" if pd.notna(cell["r_squared"]) else ""
            ctrl_row[xvar] = "Yes"
            cfe_row[xvar] = "Yes"
            yfe_row[xvar] = "Yes"

        alt_x_table_rows.extend([
            coef_row,
            se_row,
            n_row,
            country_row,
            year_row,
            r2_row,
            ctrl_row,
            cfe_row,
            yfe_row,
            {"outcome_block": "", "stat": "", **{x: "" for x in available_alt_x}}
        ])

    alt_x_main_table_df = pd.DataFrame(alt_x_table_rows)

    print("\n" + "=" * 70)
    print("替换核心解释变量主表")
    print("=" * 70)
    print(alt_x_main_table_df)


    safe_to_csv(alt_x_results_df, ALT_X_RESULTS_PATH, "替换核心解释变量稳健性结果")
    safe_to_csv(alt_x_main_table_df, ALT_X_MAIN_TABLE_PATH, "替换核心解释变量主表")