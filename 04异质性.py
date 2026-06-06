import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DATA_PATH = r"D:/课程/论文/投稿论文/数据/panel_before_imputation_2006_2023.csv"
OUTPUT_DIR = r"D:/课程/论文/投稿论文/数据/heterogeneity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: str, label: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{label}已保存：{path}")

def safe_to_excel(df: pd.DataFrame, path: str, label: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    try:
        df.to_excel(path, index=False)
        print(f"{label}已保存：{path}")
    except Exception as e:
        print(f"{label}写入 Excel 失败：{e}")

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


df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

print("\n" + "=" * 80)
print("读取数据完成")
print("=" * 80)
print("数据维度：", df.shape)
print("国家数：", df["iso3"].nunique() if "iso3" in df.columns else "缺少 iso3")
print("年份数：", df["year"].nunique() if "year" in df.columns else "缺少年份")


if "secure_employment" not in df.columns and "female_vulnerable_emp" in df.columns:
    df["secure_employment"] = 100 - df["female_vulnerable_emp"]
    print("已自动构造 secure_employment = 100 - female_vulnerable_emp")


required_common_cols = ["iso3", "year", "internet_use"]
missing_common = [c for c in required_common_cols if c not in df.columns]
if missing_common:
    raise ValueError(f"缺少必要列：{missing_common}")

# 因变量
OUTCOMES = [
    "secure_employment",
    "female_vulnerable_emp",
    "female_lfp",
    "female_employment_rate",
    "female_unemployment"
]
OUTCOMES = [v for v in OUTCOMES if v in df.columns]

if len(OUTCOMES) == 0:
    raise ValueError("没有可用的因变量。请检查数据。")

print("\n可用因变量：", OUTCOMES)

# 控制变量
CONTROLS_A = ["log_gdp_pc_ppp", "fertility", "urbanization", "service_share"]
CONTROLS_B = ["log_gdp_pc_ppp", "fertility", "urbanization", "service_share", "trade_open", "wbl_index"]



def make_baseline_high_group(
    df_input: pd.DataFrame,
    baseline_var: str,
    start_year: int = 2006,
    end_year: int = 2008,
    id_col: str = "iso3",
    time_col: str = "year",
    high_name: str = None
):
    """
    用国家层面的基期平均值（2006-2008）构造高/低组
    """
    if baseline_var not in df_input.columns:
        raise ValueError(f"数据中不存在分组变量：{baseline_var}")

    if high_name is None:
        high_name = f"high_{baseline_var}"

    base = (
        df_input.loc[df_input[time_col].between(start_year, end_year), [id_col, baseline_var]]
        .dropna()
        .groupby(id_col, as_index=False)[baseline_var]
        .mean()
        .rename(columns={baseline_var: f"{baseline_var}_base"})
    )

    if base.empty:
        raise ValueError(f"{baseline_var} 在 {start_year}-{end_year} 没有可用观测，无法构造基期组。")

    median_val = base[f"{baseline_var}_base"].median()
    base[high_name] = (base[f"{baseline_var}_base"] > median_val).astype(int)

    out = df_input.merge(base, on=id_col, how="left")

    print("\n" + "-" * 60)
    print(f"基期分组完成：{baseline_var}")
    print(f"基期年份：{start_year}-{end_year}")
    print(f"中位数：{median_val:.4f}")
    print("有基期值的国家数：", base[id_col].nunique())
    print(f"高组国家数：{base.loc[base[high_name] == 1, id_col].nunique()}")
    print(f"低组国家数：{base.loc[base[high_name] == 0, id_col].nunique()}")

    return out, median_val, base


# 异质性回归

def run_hetero_twfe(
    df_input: pd.DataFrame,
    y_var: str,
    x_var: str,
    high_var: str,
    controls: list,
    sample_flag: str,
    heterogeneity_type: str,
    baseline_var: str,
    baseline_median: float
):
    """
    TWFE:
    y ~ internet_use + internet_use*high_group + controls + C(iso3) + C(year)
    """
    needed_cols = [y_var, x_var, high_var, "iso3", "year", sample_flag] + controls
    missing_cols = [c for c in needed_cols if c not in df_input.columns]
    if missing_cols:
        print(f"[跳过] {heterogeneity_type} - {y_var}：缺少变量 {missing_cols}")
        return None, None

    df_use = df_input.loc[df_input[sample_flag] == 1].copy()

    interaction_term = f"{x_var}_X_{high_var}"
    df_use[interaction_term] = df_use[x_var] * df_use[high_var]

    use_cols = [y_var, x_var, high_var, interaction_term, "iso3", "year"] + controls
    reg_df = df_use[use_cols].dropna().copy()

    print("\n" + "=" * 80)
    print(f"异质性：{heterogeneity_type} | 因变量：{y_var}")
    print("=" * 80)
    print("sample_flag：", sample_flag)
    print("筛样后观测数：", len(df_use))
    print("dropna后观测数：", len(reg_df))
    print("国家数：", reg_df["iso3"].nunique() if not reg_df.empty else 0)
    print("年份数：", reg_df["year"].nunique() if not reg_df.empty else 0)

    if reg_df.empty:
        print(f"[跳过] {heterogeneity_type} - {y_var}：回归样本为空")
        return None, None

    if reg_df[high_var].nunique() < 2:
        print(f"[跳过] {heterogeneity_type} - {y_var}：{high_var} 在回归样本中没有组间变化")
        return None, None

    rhs = f"{x_var} + {interaction_term}"
    if len(controls) > 0:
        rhs += " + " + " + ".join(controls)

    formula = f"{y_var} ~ {rhs} + C(iso3) + C(year)"

    print("回归公式：")
    print(formula)

    model = smf.ols(formula=formula, data=reg_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["iso3"]}
    )

    # 高组总效应 = beta_x + beta_interaction
    test_high = model.t_test(f"{x_var} + {interaction_term} = 0")

    # 修复 NumPy 1.25 的 DeprecationWarning
    coef_high_total = np.asarray(test_high.effect).reshape(-1)[0]
    se_high_total = np.asarray(test_high.sd).reshape(-1)[0]
    p_high_total = np.asarray(test_high.pvalue).reshape(-1)[0]

    result_row = {
        "heterogeneity_type": heterogeneity_type,
        "baseline_var": baseline_var,
        "baseline_median": float(baseline_median),
        "sample_flag": sample_flag,
        "outcome": y_var,
        "group_var": high_var,
        "coef_low_group": model.params.get(x_var, np.nan),
        "se_low_group": model.bse.get(x_var, np.nan),
        "p_low_group": model.pvalues.get(x_var, np.nan),
        "coef_diff_high_minus_low": model.params.get(interaction_term, np.nan),
        "se_diff_high_minus_low": model.bse.get(interaction_term, np.nan),
        "p_diff_high_minus_low": model.pvalues.get(interaction_term, np.nan),
        "coef_high_group_total": float(coef_high_total),
        "se_high_group_total": float(se_high_total),
        "p_high_group_total": float(p_high_total),
        "n_obs": int(model.nobs),
        "n_country": int(reg_df["iso3"].nunique()),
        "n_year": int(reg_df["year"].nunique()),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "formula": formula
    }

    focus_print = pd.DataFrame([
        {
            "term": "Low group effect",
            "coef": result_row["coef_low_group"],
            "std_err": result_row["se_low_group"],
            "p_value": result_row["p_low_group"]
        },
        {
            "term": "High - Low difference",
            "coef": result_row["coef_diff_high_minus_low"],
            "std_err": result_row["se_diff_high_minus_low"],
            "p_value": result_row["p_diff_high_minus_low"]
        },
        {
            "term": "High group total effect",
            "coef": result_row["coef_high_group_total"],
            "std_err": result_row["se_high_group_total"],
            "p_value": result_row["p_high_group_total"]
        }
    ])

    print("\n核心结果：")
    print(focus_print.round(4).to_string(index=False))

    return result_row, model


# 批量运行某一组异质性

def run_one_heterogeneity_bundle(
    df_input: pd.DataFrame,
    heterogeneity_type: str,
    baseline_var: str,
    high_name: str,
    sample_flag: str,
    controls: list,
    outcomes: list,
    start_year: int = 2006,
    end_year: int = 2008
):
    work_df, median_val, base_df = make_baseline_high_group(
        df_input=df_input,
        baseline_var=baseline_var,
        start_year=start_year,
        end_year=end_year,
        id_col="iso3",
        time_col="year",
        high_name=high_name
    )

    result_rows = []
    model_store = {}

    for y in outcomes:
        row, model = run_hetero_twfe(
            df_input=work_df,
            y_var=y,
            x_var="internet_use",
            high_var=high_name,
            controls=controls,
            sample_flag=sample_flag,
            heterogeneity_type=heterogeneity_type,
            baseline_var=baseline_var,
            baseline_median=median_val
        )
        if row is not None:
            result_rows.append(row)
            model_store[y] = model

    results_df = pd.DataFrame(result_rows)

    if results_df.empty:
        print(f"\n{heterogeneity_type} 没有成功生成任何结果。")
        return work_df, base_df, results_df, model_store

    # 加星号
    results_df["low_group_star"] = results_df.apply(
        lambda r: add_stars(r["coef_low_group"], r["p_low_group"]), axis=1
    )
    results_df["diff_star"] = results_df.apply(
        lambda r: add_stars(r["coef_diff_high_minus_low"], r["p_diff_high_minus_low"]), axis=1
    )
    results_df["high_group_total_star"] = results_df.apply(
        lambda r: add_stars(r["coef_high_group_total"], r["p_high_group_total"]), axis=1
    )

    results_df["se_low_group_fmt"] = results_df["se_low_group"].apply(lambda v: f"({v:.4f})" if pd.notna(v) else "")
    results_df["se_diff_fmt"] = results_df["se_diff_high_minus_low"].apply(lambda v: f"({v:.4f})" if pd.notna(v) else "")
    results_df["se_high_total_fmt"] = results_df["se_high_group_total"].apply(lambda v: f"({v:.4f})" if pd.notna(v) else "")

    print("\n" + "#" * 80)
    print(f"{heterogeneity_type} 结果汇总")
    print("#" * 80)
    print(
        results_df[
            [
                "outcome",
                "low_group_star",
                "se_low_group_fmt",
                "diff_star",
                "se_diff_fmt",
                "high_group_total_star",
                "se_high_total_fmt",
                "n_obs",
                "n_country",
                "r_squared"
            ]
        ].to_string(index=False)
    )

    return work_df, base_df, results_df, model_store


#  整理展示表

def build_presentation_table(results_df: pd.DataFrame, heterogeneity_label: str):
    if results_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in results_df.iterrows():
        outcome = row["outcome"]

        coef_low_row = {
            "outcome": outcome,
            "stat": "Low group effect",
            heterogeneity_label: row["low_group_star"]
        }
        se_low_row = {
            "outcome": outcome,
            "stat": "Std. Err. (Low group)",
            heterogeneity_label: row["se_low_group_fmt"]
        }

        coef_diff_row = {
            "outcome": outcome,
            "stat": "High - Low difference",
            heterogeneity_label: row["diff_star"]
        }
        se_diff_row = {
            "outcome": outcome,
            "stat": "Std. Err. (Difference)",
            heterogeneity_label: row["se_diff_fmt"]
        }

        coef_high_row = {
            "outcome": outcome,
            "stat": "High group total effect",
            heterogeneity_label: row["high_group_total_star"]
        }
        se_high_row = {
            "outcome": outcome,
            "stat": "Std. Err. (High total)",
            heterogeneity_label: row["se_high_total_fmt"]
        }

        n_row = {
            "outcome": outcome,
            "stat": "N",
            heterogeneity_label: int(row["n_obs"]) if pd.notna(row["n_obs"]) else ""
        }
        country_row = {
            "outcome": outcome,
            "stat": "Countries",
            heterogeneity_label: int(row["n_country"]) if pd.notna(row["n_country"]) else ""
        }
        year_row = {
            "outcome": outcome,
            "stat": "Years",
            heterogeneity_label: int(row["n_year"]) if pd.notna(row["n_year"]) else ""
        }
        r2_row = {
            "outcome": outcome,
            "stat": "R2",
            heterogeneity_label: f"{row['r_squared']:.4f}" if pd.notna(row["r_squared"]) else ""
        }

        rows.extend([
            coef_low_row, se_low_row,
            coef_diff_row, se_diff_row,
            coef_high_row, se_high_row,
            n_row, country_row, year_row, r2_row
        ])

    return pd.DataFrame(rows)


# 三组异质性配置

heterogeneity_configs = [
    {
        "heterogeneity_type": "female_education",
        "heterogeneity_label": "Female education high vs low",
        "baseline_var": "female_tertiary",
        "high_name": "high_female_edu",
        "sample_flag": "sample_C_complete",
        "controls": CONTROLS_B
    },
    {
        "heterogeneity_type": "service_structure",
        "heterogeneity_label": "Service share high vs low",
        "baseline_var": "service_share",
        "high_name": "high_service",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    },
    {
        "heterogeneity_type": "institution_wbl",
        "heterogeneity_label": "Institutional environment high vs low",
        "baseline_var": "wbl_index",
        "high_name": "high_wbl_env",
        "sample_flag": "sample_B_complete",
        "controls": CONTROLS_B
    }
]


#  批量运行三组异质性

all_results = []
all_presentation_tables = []

for cfg in heterogeneity_configs:
    print("\n" + "\n" + "=" * 100)
    print(f"开始运行异质性：{cfg['heterogeneity_type']}")
    print("=" * 100)

    try:
        work_df, base_df, res_df, model_store = run_one_heterogeneity_bundle(
            df_input=df,
            heterogeneity_type=cfg["heterogeneity_type"],
            baseline_var=cfg["baseline_var"],
            high_name=cfg["high_name"],
            sample_flag=cfg["sample_flag"],
            controls=cfg["controls"],
            outcomes=OUTCOMES,
            start_year=2006,
            end_year=2008
        )

        # 保存国家层面的基期分组表
        baseline_path_csv = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_baseline_groups.csv")
        baseline_path_xlsx = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_baseline_groups.xlsx")
        safe_to_csv(base_df, baseline_path_csv, f"{cfg['heterogeneity_type']} 基期分组表")
        safe_to_excel(base_df, baseline_path_xlsx, f"{cfg['heterogeneity_type']} 基期分组表")

        # 保存长表结果
        result_path_csv = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_results_long.csv")
        result_path_xlsx = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_results_long.xlsx")

        if not res_df.empty:
            safe_to_csv(res_df, result_path_csv, f"{cfg['heterogeneity_type']} 长表结果")
            safe_to_excel(res_df, result_path_xlsx, f"{cfg['heterogeneity_type']} 长表结果")
            all_results.append(res_df)

            # 展示表
            pres_df = build_presentation_table(res_df, cfg["heterogeneity_label"])
            pres_path_csv = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_presentation_table.csv")
            pres_path_xlsx = os.path.join(OUTPUT_DIR, f"{cfg['heterogeneity_type']}_presentation_table.xlsx")
            safe_to_csv(pres_df, pres_path_csv, f"{cfg['heterogeneity_type']} 展示表")
            safe_to_excel(pres_df, pres_path_xlsx, f"{cfg['heterogeneity_type']} 展示表")
            all_presentation_tables.append(pres_df)

        else:
            print(f"{cfg['heterogeneity_type']} 没有生成有效结果。")

    except Exception as e:
        print(f"\n[错误] {cfg['heterogeneity_type']} 运行失败：{e}")


#  合并总表

if len(all_results) > 0:
    final_results_df = pd.concat(all_results, axis=0, ignore_index=True)

    final_csv = os.path.join(OUTPUT_DIR, "heterogeneity_all_results_long.csv")
    final_xlsx = os.path.join(OUTPUT_DIR, "heterogeneity_all_results_long.xlsx")

    safe_to_csv(final_results_df, final_csv, "异质性总长表")
    safe_to_excel(final_results_df, final_xlsx, "异质性总长表")

    print("\n" + "=" * 100)
    print("全部异质性长表结果汇总")
    print("=" * 100)
    print(
        final_results_df[
            [
                "heterogeneity_type",
                "outcome",
                "low_group_star",
                "diff_star",
                "high_group_total_star",
                "n_obs",
                "n_country",
                "r_squared"
            ]
        ].to_string(index=False)
    )
else:
    print("\n没有成功生成任何异质性长表结果。")

if len(all_presentation_tables) > 0:
    final_presentation_df = pd.concat(all_presentation_tables, axis=0, ignore_index=True)

    final_pres_csv = os.path.join(OUTPUT_DIR, "heterogeneity_all_presentation_tables.csv")
    final_pres_xlsx = os.path.join(OUTPUT_DIR, "heterogeneity_all_presentation_tables.xlsx")

    safe_to_csv(final_presentation_df, final_pres_csv, "异质性总展示表")
    safe_to_excel(final_presentation_df, final_pres_xlsx, "异质性总展示表")

    print("\n" + "=" * 100)
    print("全部异质性展示表汇总")
    print("=" * 100)
    print(final_presentation_df.to_string(index=False))
else:
    print("\n没有成功生成任何异质性展示表。")

print("\n" + "=" * 100)
print("04异质性.py 运行结束")
print("=" * 100)


#  机制部分（新增）
#     M1: 转型摩擦机制
#     M2: 机制支持性异质性整理


MECH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mechanism_results")
os.makedirs(MECH_OUTPUT_DIR, exist_ok=True)

#工具函数：按国家生成滞后项

def add_country_lag(df_input: pd.DataFrame, var: str, lag: int = 1,
                    id_col: str = "iso3", time_col: str = "year",
                    out_name: str = None):
    """
    在国家维度上生成滞后项
    """
    if var not in df_input.columns:
        raise ValueError(f"无法生成滞后项，缺少变量：{var}")

    if out_name is None:
        out_name = f"{var}_l{lag}"

    work = df_input.sort_values([id_col, time_col]).copy()
    work[out_name] = work.groupby(id_col)[var].shift(lag)
    return work

# 如果一期滞后不存在，就自动生成
if "internet_use_l1" not in df.columns:
    df = add_country_lag(df, var="internet_use", lag=1, out_name="internet_use_l1")
    print("已自动生成滞后变量：internet_use_l1")



#机制回归（非异质性）

def run_mechanism_twfe(
    df_input: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list,
    sample_flag: str,
    mechanism_block: str,
    mechanism_label: str
):
    """
    标准 TWFE:
    y ~ x + controls + C(iso3) + C(year)
    """
    needed_cols = [y_var, x_var, "iso3", "year", sample_flag] + controls
    missing_cols = [c for c in needed_cols if c not in df_input.columns]
    if missing_cols:
        print(f"[跳过] {mechanism_label} - {y_var}：缺少变量 {missing_cols}")
        return None, None

    df_use = df_input.loc[df_input[sample_flag] == 1].copy()
    reg_df = df_use[[y_var, x_var, "iso3", "year"] + controls].dropna().copy()

    print("\n" + "=" * 80)
    print(f"机制回归：{mechanism_label} | 因变量：{y_var}")
    print("=" * 80)
    print("sample_flag：", sample_flag)
    print("筛样后观测数：", len(df_use))
    print("dropna后观测数：", len(reg_df))
    print("国家数：", reg_df["iso3"].nunique() if not reg_df.empty else 0)
    print("年份数：", reg_df["year"].nunique() if not reg_df.empty else 0)

    if reg_df.empty:
        print(f"[跳过] {mechanism_label} - {y_var}：回归样本为空")
        return None, None

    rhs = x_var
    if len(controls) > 0:
        rhs += " + " + " + ".join(controls)

    formula = f"{y_var} ~ {rhs} + C(iso3) + C(year)"

    print("回归公式：")
    print(formula)

    model = smf.ols(formula=formula, data=reg_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df["iso3"]}
    )

    row = {
        "mechanism_block": mechanism_block,
        "mechanism_label": mechanism_label,
        "outcome": y_var,
        "x_var": x_var,
        "sample_flag": sample_flag,
        "coef": model.params.get(x_var, np.nan),
        "std_err": model.bse.get(x_var, np.nan),
        "p_value": model.pvalues.get(x_var, np.nan),
        "n_obs": int(model.nobs),
        "n_country": int(reg_df["iso3"].nunique()),
        "n_year": int(reg_df["year"].nunique()),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "formula": formula
    }

    focus_print = pd.DataFrame([{
        "term": x_var,
        "coef": row["coef"],
        "std_err": row["std_err"],
        "p_value": row["p_value"]
    }])

    print("\n核心结果：")
    print(focus_print.round(4).to_string(index=False))

    return row, model


#  运行 M1：转型摩擦机制表

mechanism_specs = [
    {
        "mechanism_block": "M1_transitional_friction",
        "mechanism_label": "Current friction: internet use -> female unemployment",
        "y_var": "female_unemployment",
        "x_var": "internet_use",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    },
    {
        "mechanism_block": "M1_lagged_quality_upgrading",
        "mechanism_label": "Lagged upgrading: internet use(lag1) -> secure employment",
        "y_var": "secure_employment",
        "x_var": "internet_use_l1",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    },
    {
        "mechanism_block": "M1_lagged_quality_upgrading",
        "mechanism_label": "Lagged upgrading: internet use(lag1) -> female vulnerable employment",
        "y_var": "female_vulnerable_emp",
        "x_var": "internet_use_l1",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    },
    {
        "mechanism_block": "M1_lagged_quantity_placebo",
        "mechanism_label": "Lagged quantity check: internet use(lag1) -> female labor force participation",
        "y_var": "female_lfp",
        "x_var": "internet_use_l1",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    },
    {
        "mechanism_block": "M1_lagged_quantity_placebo",
        "mechanism_label": "Lagged quantity check: internet use(lag1) -> female employment rate",
        "y_var": "female_employment_rate",
        "x_var": "internet_use_l1",
        "sample_flag": "sample_A_complete",
        "controls": CONTROLS_A
    }
]

mechanism_rows = []
mechanism_model_store = {}

print("\n" + "\n" + "=" * 100)
print("开始运行机制部分：M1 转型摩擦机制")
print("=" * 100)

for spec in mechanism_specs:
    try:
        row, model = run_mechanism_twfe(
            df_input=df,
            y_var=spec["y_var"],
            x_var=spec["x_var"],
            controls=spec["controls"],
            sample_flag=spec["sample_flag"],
            mechanism_block=spec["mechanism_block"],
            mechanism_label=spec["mechanism_label"]
        )
        if row is not None:
            mechanism_rows.append(row)
            mechanism_model_store[f"{spec['x_var']}__{spec['y_var']}"] = model
    except Exception as e:
        print(f"[错误] 机制回归失败：{spec['mechanism_label']} | {e}")

mechanism_df = pd.DataFrame(mechanism_rows)

if not mechanism_df.empty:
    mechanism_df["coef_star"] = mechanism_df.apply(
        lambda r: add_stars(r["coef"], r["p_value"]), axis=1
    )
    mechanism_df["se_fmt"] = mechanism_df["std_err"].apply(
        lambda v: f"({v:.4f})" if pd.notna(v) else ""
    )

    print("\n" + "#" * 80)
    print("M1 转型摩擦机制结果汇总")
    print("#" * 80)
    print(
        mechanism_df[
            [
                "mechanism_block",
                "outcome",
                "x_var",
                "coef_star",
                "se_fmt",
                "n_obs",
                "n_country",
                "r_squared"
            ]
        ].to_string(index=False)
    )

    mech_long_csv = os.path.join(MECH_OUTPUT_DIR, "mechanism_M1_transitional_friction_long.csv")
    mech_long_xlsx = os.path.join(MECH_OUTPUT_DIR, "mechanism_M1_transitional_friction_long.xlsx")
    safe_to_csv(mechanism_df, mech_long_csv, "M1 机制长表")
    safe_to_excel(mechanism_df, mech_long_xlsx, "M1 机制长表")

    # M1 展示表
    mech_pres_rows = []
    for _, row in mechanism_df.iterrows():
        mech_pres_rows.extend([
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": row["x_var"],
                "value": row["coef_star"]
            },
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": "Std. Err.",
                "value": row["se_fmt"]
            },
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": "N",
                "value": int(row["n_obs"]) if pd.notna(row["n_obs"]) else ""
            },
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": "Countries",
                "value": int(row["n_country"]) if pd.notna(row["n_country"]) else ""
            },
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": "Years",
                "value": int(row["n_year"]) if pd.notna(row["n_year"]) else ""
            },
            {
                "mechanism_block": row["mechanism_block"],
                "outcome": row["outcome"],
                "stat": "R2",
                "value": f"{row['r_squared']:.4f}" if pd.notna(row["r_squared"]) else ""
            }
        ])

    mechanism_pres_df = pd.DataFrame(mech_pres_rows)

    mech_pres_csv = os.path.join(MECH_OUTPUT_DIR, "mechanism_M1_transitional_friction_presentation.csv")
    mech_pres_xlsx = os.path.join(MECH_OUTPUT_DIR, "mechanism_M1_transitional_friction_presentation.xlsx")
    safe_to_csv(mechanism_pres_df, mech_pres_csv, "M1 机制展示表")
    safe_to_excel(mechanism_pres_df, mech_pres_xlsx, "M1 机制展示表")
else:
    print("\nM1 没有成功生成有效结果。")


#M2：用现有异质性结果整理机制支持表

print("\n" + "\n" + "=" * 100)
print("开始整理机制部分：M2 机制支持性异质性表")
print("=" * 100)

if "final_results_df" in locals() and not final_results_df.empty:
    mechanism_map = {
        "female_education": "Human capital complementarity",
        "service_structure": "Sectoral absorption capacity"
    }

    m2_df = final_results_df[
        final_results_df["heterogeneity_type"].isin(mechanism_map.keys())
    ].copy()

    if not m2_df.empty:
        m2_df["mechanism_channel"] = m2_df["heterogeneity_type"].map(mechanism_map)

        # 只保留机制解释最需要的列
        m2_keep_cols = [
            "mechanism_channel",
            "heterogeneity_type",
            "outcome",
            "low_group_star",
            "se_low_group_fmt",
            "diff_star",
            "se_diff_fmt",
            "high_group_total_star",
            "se_high_total_fmt",
            "n_obs",
            "n_country",
            "n_year",
            "r_squared",
            "formula"
        ]
        m2_df = m2_df[m2_keep_cols].copy()

        print("\n" + "#" * 80)
        print("M2 机制支持性异质性结果汇总")
        print("#" * 80)
        print(
            m2_df[
                [
                    "mechanism_channel",
                    "outcome",
                    "low_group_star",
                    "diff_star",
                    "high_group_total_star",
                    "n_obs",
                    "n_country",
                    "r_squared"
                ]
            ].to_string(index=False)
        )

        m2_long_csv = os.path.join(MECH_OUTPUT_DIR, "mechanism_M2_supporting_heterogeneity_long.csv")
        m2_long_xlsx = os.path.join(MECH_OUTPUT_DIR, "mechanism_M2_supporting_heterogeneity_long.xlsx")
        safe_to_csv(m2_df, m2_long_csv, "M2 机制支持长表")
        safe_to_excel(m2_df, m2_long_xlsx, "M2 机制支持长表")

        # M2 展示表
        m2_pres_rows = []
        for _, row in m2_df.iterrows():
            m2_pres_rows.extend([
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Low group effect",
                    "value": row["low_group_star"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Std. Err. (Low group)",
                    "value": row["se_low_group_fmt"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "High - Low difference",
                    "value": row["diff_star"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Std. Err. (Difference)",
                    "value": row["se_diff_fmt"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "High group total effect",
                    "value": row["high_group_total_star"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Std. Err. (High total)",
                    "value": row["se_high_total_fmt"]
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "N",
                    "value": int(row["n_obs"]) if pd.notna(row["n_obs"]) else ""
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Countries",
                    "value": int(row["n_country"]) if pd.notna(row["n_country"]) else ""
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "Years",
                    "value": int(row["n_year"]) if pd.notna(row["n_year"]) else ""
                },
                {
                    "mechanism_channel": row["mechanism_channel"],
                    "outcome": row["outcome"],
                    "stat": "R2",
                    "value": f"{row['r_squared']:.4f}" if pd.notna(row["r_squared"]) else ""
                }
            ])

        m2_pres_df = pd.DataFrame(m2_pres_rows)

        m2_pres_csv = os.path.join(MECH_OUTPUT_DIR, "mechanism_M2_supporting_heterogeneity_presentation.csv")
        m2_pres_xlsx = os.path.join(MECH_OUTPUT_DIR, "mechanism_M2_supporting_heterogeneity_presentation.xlsx")
        safe_to_csv(m2_pres_df, m2_pres_csv, "M2 机制支持展示表")
        safe_to_excel(m2_pres_df, m2_pres_xlsx, "M2 机制支持展示表")

    else:
        print("M2 整理失败：final_results_df 中没有 female_education / service_structure 结果。")
else:
    print("M2 无法整理：前面没有生成 final_results_df。")



# 机制汇总

print("\n" + "=" * 100)
print("机制部分运行结束")
print("=" * 100)