import os
import warnings
warnings.filterwarnings("ignore")


SHOW_RESULT_IN_OUTPUT = True
SHOW_PLOTS_IN_OUTPUT = False   

SAVE_SUMMARY_CSV = True
SAVE_FEATURE_IMPORTANCE_CSV = True
SAVE_DECILE_CSV = True
SAVE_PANEL_CSV = False        
SAVE_BINNED_CSV = False      
SAVE_PLOTS = True
SAVE_CATE_HIST_PLOT = False   
SAVE_EXTENDED_PLOTS = False    


RUN_COUNTRY_GRADIENT = True
RUN_SEED_STABILITY = True
SAVE_COUNTRY_GRADIENT_CSV = True
SAVE_COUNTRY_GRADIENT_SUMMARY_CSV = True
SAVE_STABILITY_CSV = True
SAVE_STABILITY_LONG_CSV = False   
STABILITY_ONLY_FOR_SECURE = True 
STABILITY_SEEDS = list(range(2026, 2036))  

import matplotlib
if not SHOW_PLOTS_IN_OUTPUT:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML



DATA_PATH = r"D:/课程/论文/投稿论文/数据/panel_before_imputation_2006_2023.csv"
OUTPUT_ROOT = r"D:/课程/论文/投稿论文/数据/ml_results_dual_track_simplified"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

RANDOM_STATE = 2026

ALL_BASELINE_VARS = [
    "female_tertiary",
    "service_share",
    "wbl_index",
    "log_gdp_pc_ppp",
    "urbanization",
    "fertility",
    "trade_open"
]

BASE_CONTROLS = [
    "log_gdp_pc_ppp",
    "fertility",
    "urbanization",
    "service_share",
    "trade_open",
    "wbl_index"
]

FEATURE_LABELS = {
    "female_tertiary_base": "Female tertiary enrollment (baseline)",
    "service_share_base": "Service share (baseline)",
    "wbl_index_base": "WBL index (baseline)",
    "log_gdp_pc_ppp_base": "Log GDP per capita PPP (baseline)",
    "urbanization_base": "Urbanization (baseline)",
    "fertility_base": "Fertility (baseline)",
    "trade_open_base": "Trade openness (baseline)"
}

MODEL_SPECS = {
    "main_text": {
        "label": "Main-text ML",
        "sample_flag": "sample_C_complete",
        "outcomes": ["secure_employment"],
        "hetero_feature_bases": [
            "female_tertiary",
            "service_share",
            "wbl_index"
        ],
        "controls": BASE_CONTROLS,
        "plot_features": [
            "female_tertiary_base",
            "service_share_base",
            "wbl_index_base"
        ]
    },
    "extended": {
        "label": "Extended ML",
        "sample_flag": "sample_C_complete",
        "outcomes": ["secure_employment", "female_vulnerable_emp"],
        "hetero_feature_bases": [
            "female_tertiary",
            "service_share",
            "wbl_index",
            "log_gdp_pc_ppp",
            "urbanization",
            "fertility",
            "trade_open"
        ],
        "controls": BASE_CONTROLS,
        "plot_features": [
            "female_tertiary_base",
            "service_share_base",
            "wbl_index_base",
            "log_gdp_pc_ppp_base",
            "urbanization_base",
            "fertility_base",
            "trade_open_base"
        ]
    }
}


def safe_to_csv(df: pd.DataFrame, path: str, label: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{label}已保存：{path}")


def maybe_save_csv(df: pd.DataFrame, enabled: bool, path: str, label: str):
    if enabled:
        safe_to_csv(df, path, label)


def finalize_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if SHOW_PLOTS_IN_OUTPUT:
        plt.show()
    plt.close()
    print(f"图已保存：{path}")


def make_country_baseline(
    df_input: pd.DataFrame,
    vars_for_base: list,
    start_year: int = 2006,
    end_year: int = 2008,
    id_col: str = "iso3",
    time_col: str = "year"
) -> pd.DataFrame:
    use_cols = [id_col, time_col] + vars_for_base
    tmp = df_input[use_cols].copy()
    tmp = tmp.loc[tmp[time_col].between(start_year, end_year)].copy()

    base = tmp.groupby(id_col)[vars_for_base].mean().reset_index()
    base = base.rename(columns={c: f"{c}_base" for c in vars_for_base})
    return base


def residualize_two_way_fe(
    df_input: pd.DataFrame,
    y_var: str,
    rhs_vars: list,
    entity_col: str = "iso3",
    time_col: str = "year"
) -> pd.Series:
    needed = [entity_col, time_col, y_var] + rhs_vars
    reg_df = df_input[needed].dropna().copy()
    reg_df = reg_df.set_index([entity_col, time_col])

    y = reg_df[y_var]
    X = reg_df[rhs_vars]

    model = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    res = model.fit()
    return pd.Series(index=reg_df.index, data=res.resids, name=f"{y_var}_resid")



def cache_key(sample_flag: str, controls: list, var_name: str) -> tuple:
    return (sample_flag, tuple(controls), var_name)


def get_treatment_residual(
    df_input: pd.DataFrame,
    treat_var: str,
    controls: list,
    sample_flag: str,
    resid_cache: dict
) -> pd.Series:
    key = cache_key(sample_flag, controls, treat_var)
    if key in resid_cache:
        return resid_cache[key]

    work = df_input.loc[df_input[sample_flag] == 1].copy()
    t_res = residualize_two_way_fe(
        df_input=work,
        y_var=treat_var,
        rhs_vars=controls,
        entity_col="iso3",
        time_col="year"
    )
    resid_cache[key] = t_res
    return t_res



def build_ml_sample(
    df_input: pd.DataFrame,
    outcome_var: str,
    treat_var: str,
    controls: list,
    hetero_features: list,
    sample_flag: str,
    resid_cache: dict
) -> pd.DataFrame:
    work = df_input.loc[df_input[sample_flag] == 1].copy()
    work = work.sort_values(["iso3", "year"]).reset_index(drop=True)

    y_res = residualize_two_way_fe(
        df_input=work,
        y_var=outcome_var,
        rhs_vars=controls,
        entity_col="iso3",
        time_col="year"
    )

    t_res = get_treatment_residual(
        df_input=df_input,
        treat_var=treat_var,
        controls=controls,
        sample_flag=sample_flag,
        resid_cache=resid_cache
    )

    work = work.set_index(["iso3", "year"])
    work = work.join(y_res, how="left")
    work = work.join(t_res, how="left")
    work = work.reset_index()

    if "country_name" not in work.columns:
        work["country_name"] = work["iso3"]

    use_cols = (
        ["iso3", "country_name", "year", outcome_var, treat_var,
         f"{outcome_var}_resid", f"{treat_var}_resid"]
        + controls
        + hetero_features
    )

    ml_df = work[use_cols].dropna().copy()

    print("\n" + "=" * 80)
    print(f"ML样本：{outcome_var}")
    print("=" * 80)
    print("观测数：", len(ml_df))
    print("国家数：", ml_df["iso3"].nunique())
    print("年份数：", ml_df["year"].nunique())

    return ml_df


#  Causal Forest

def fit_causal_forest_continuous(
    ml_df: pd.DataFrame,
    outcome_var: str,
    treat_var: str,
    hetero_features: list,
    random_state: int = RANDOM_STATE
):
    Y = ml_df[f"{outcome_var}_resid"].values
    T = ml_df[f"{treat_var}_resid"].values
    X = ml_df[hetero_features].values

    model_y = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=8,
        random_state=random_state,
        n_jobs=-1
    )

    model_t = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=8,
        random_state=random_state,
        n_jobs=-1
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=800,
        min_samples_leaf=10,
        max_depth=None,
        max_samples=0.45,
        cv=3,
        random_state=random_state,
        inference=False
    )

    cf.fit(Y=Y, T=T, X=X)
    tau_hat = cf.const_marginal_effect(X).reshape(-1)

    ml_df = ml_df.copy()
    ml_df["cate_hat"] = tau_hat

    ate_hat = float(np.mean(tau_hat))
    ate_sd = float(np.std(tau_hat, ddof=1))

    fi_df = pd.DataFrame({
        "feature": hetero_features,
        "importance": cf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return ml_df, ate_hat, ate_sd, fi_df


#  结果

def make_decile_table(ml_df: pd.DataFrame, cate_col: str = "cate_hat") -> pd.DataFrame:
    out = ml_df.copy()
    out["cate_decile"] = pd.qcut(out[cate_col], 10, labels=False, duplicates="drop") + 1
    tab = (
        out.groupby("cate_decile")
        .agg(
            n_obs=(cate_col, "size"),
            cate_mean=(cate_col, "mean"),
            cate_sd=(cate_col, "std"),
            n_country=("iso3", "nunique")
        )
        .reset_index()
    )
    return tab


def make_binned_effect_table(
    ml_df: pd.DataFrame,
    x_col: str,
    cate_col: str = "cate_hat",
    n_bins: int = 10
) -> pd.DataFrame:
    out = ml_df.copy()
    out["bin"] = pd.qcut(out[x_col], n_bins, labels=False, duplicates="drop") + 1
    tab = (
        out.groupby("bin")
        .agg(
            x_mean=(x_col, "mean"),
            cate_mean=(cate_col, "mean"),
            cate_sd=(cate_col, "std"),
            n_obs=(cate_col, "size")
        )
        .reset_index()
    )
    return tab



# 新增：Country-level moderator gradient

def make_country_level_cate_table(
    ml_df: pd.DataFrame,
    moderator: str,
    n_groups: int = 3
):
    tmp = ml_df.groupby("iso3", as_index=False).agg({
        "cate_hat": "mean",
        moderator: "first"
    }).dropna().copy()

    if tmp.empty or tmp[moderator].nunique() < 2:
        empty_tab = pd.DataFrame(columns=["group", "n_country", "moderator_mean", "cate_mean", "cate_sd"])
        return empty_tab, np.nan

    try:
        tmp["group"] = pd.qcut(tmp[moderator], n_groups, labels=False, duplicates="drop") + 1
    except ValueError:
        empty_tab = pd.DataFrame(columns=["group", "n_country", "moderator_mean", "cate_mean", "cate_sd"])
        return empty_tab, np.nan

    tab = (
        tmp.groupby("group")
        .agg(
            n_country=("iso3", "nunique"),
            moderator_mean=(moderator, "mean"),
            cate_mean=("cate_hat", "mean"),
            cate_sd=("cate_hat", "std")
        )
        .reset_index()
    )

    if len(tab) >= 2:
        high_low_gap = (
            tab.loc[tab["group"] == tab["group"].max(), "cate_mean"].iloc[0]
            - tab.loc[tab["group"] == tab["group"].min(), "cate_mean"].iloc[0]
        )
    else:
        high_low_gap = np.nan

    return tab, float(high_low_gap) if pd.notna(high_low_gap) else np.nan


def print_country_gradient_result(spec_name: str, outcome_var: str, moderator: str,
                                  country_tab: pd.DataFrame, high_low_gap: float):
    if not SHOW_RESULT_IN_OUTPUT:
        return

    print("\n" + "-" * 80)
    print(f"[{spec_name} | {outcome_var}] country-level moderator gradient: {moderator}")
    print("-" * 80)
    if country_tab.empty:
        print("结果为空，可能是分组失败或 moderator 取值变化过少。")
    else:
        print(country_tab.to_string(index=False))
        print(f"High-Low gap: {high_low_gap:.6f}" if pd.notna(high_low_gap) else "High-Low gap: NaN")


#  新增：Feature-ranking stability

def run_seed_stability(
    ml_df: pd.DataFrame,
    outcome_var: str,
    treat_var: str,
    hetero_features: list,
    seeds=None
):
    if seeds is None:
        seeds = STABILITY_SEEDS

    rows = []

    Y = ml_df[f"{outcome_var}_resid"].values
    T = ml_df[f"{treat_var}_resid"].values
    X = ml_df[hetero_features].values

    for seed in seeds:
        model_y = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=-1
        )

        model_t = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=-1
        )

        cf = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=800,
            min_samples_leaf=10,
            max_depth=None,
            max_samples=0.45,
            cv=3,
            random_state=seed,
            inference=False
        )

        cf.fit(Y=Y, T=T, X=X)

        fi = pd.DataFrame({
            "feature": hetero_features,
            "importance": cf.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        fi["rank"] = np.arange(1, len(fi) + 1)
        fi["seed"] = seed
        rows.append(fi)

    long_df = pd.concat(rows, ignore_index=True)

    stability_df = (
        long_df.groupby("feature")
        .agg(
            mean_rank=("rank", "mean"),
            top1_freq=("rank", lambda x: np.mean(x == 1)),
            top3_freq=("rank", lambda x: np.mean(x <= 3)),
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std")
        )
        .reset_index()
        .sort_values(["top3_freq", "top1_freq", "mean_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    return long_df, stability_df


def print_stability_result(spec_name: str, outcome_var: str, stability_df: pd.DataFrame):
    if not SHOW_RESULT_IN_OUTPUT:
        return

    print("\n" + "-" * 80)
    print(f"[{spec_name} | {outcome_var}] feature-ranking stability")
    print("-" * 80)
    print(stability_df.to_string(index=False))



# 输出

def print_core_results(spec_name: str, outcome_var: str, ate_hat: float, ate_sd: float,
                       fi_df: pd.DataFrame, decile_df: pd.DataFrame):
    if not SHOW_RESULT_IN_OUTPUT:
        return

    print("\n" + "-" * 80)
    print(f"[{spec_name} | {outcome_var}] 机器学习核心结果")
    print("-" * 80)
    print(f"ATE-like mean: {ate_hat:.6f}")
    print(f"ATE-like sd  : {ate_sd:.6f}")
    print("\nTop 5 features:")
    print(fi_df.head(5).to_string(index=False))
    print("\nCATE decile table:")
    print(decile_df.to_string(index=False))


# 图

def plot_hist_cate(ml_df: pd.DataFrame, outcome_var: str, output_dir: str, prefix: str):
    plt.figure(figsize=(8, 5))
    plt.hist(ml_df["cate_hat"], bins=30)
    plt.axvline(ml_df["cate_hat"].mean(), linestyle="--")
    plt.title(f"CATE distribution: {outcome_var}")
    plt.xlabel("Estimated marginal effect of internet_use")
    plt.ylabel("Frequency")
    path = os.path.join(output_dir, f"{prefix}_{outcome_var}_cate_distribution.png")
    finalize_plot(path)


def plot_feature_importance(fi_df: pd.DataFrame, outcome_var: str, output_dir: str, prefix: str):
    tmp = fi_df.sort_values("importance", ascending=True).copy()
    tmp["feature_label"] = tmp["feature"].map(lambda x: FEATURE_LABELS.get(x, x))

    plt.figure(figsize=(8, 5))
    plt.barh(tmp["feature_label"], tmp["importance"])
    plt.title(f"Feature importance: {outcome_var}")
    plt.xlabel("Importance")
    plt.ylabel("")
    path = os.path.join(output_dir, f"{prefix}_{outcome_var}_feature_importance.png")
    finalize_plot(path)


def plot_binned_curve(
    tab: pd.DataFrame,
    x_label: str,
    outcome_var: str,
    file_stub: str,
    output_dir: str,
    prefix: str
):
    plt.figure(figsize=(7, 5))
    plt.plot(tab["x_mean"], tab["cate_mean"], marker="o")
    plt.title(f"CATE vs {x_label}: {outcome_var}")
    plt.xlabel(x_label)
    plt.ylabel("Mean estimated marginal effect")
    path = os.path.join(output_dir, f"{prefix}_{outcome_var}_{file_stub}_binned_curve.png")
    finalize_plot(path)


#  单个规格运行器

def run_one_spec(
    df_input: pd.DataFrame,
    spec_name: str,
    spec_config: dict,
    resid_cache: dict
):
    spec_dir = os.path.join(OUTPUT_ROOT, spec_name)
    os.makedirs(spec_dir, exist_ok=True)

    label = spec_config["label"]
    sample_flag = spec_config["sample_flag"]
    outcomes = spec_config["outcomes"]
    controls = spec_config["controls"]
    hetero_features = [f"{v}_base" for v in spec_config["hetero_feature_bases"]]
    plot_features = spec_config["plot_features"]

    print("\n" + "#" * 100)
    print(f"开始运行规格：{spec_name} | {label}")
    print("#" * 100)
    print("sample_flag：", sample_flag)
    print("outcomes：", outcomes)
    print("hetero_features：", hetero_features)

    summary_rows = []
    all_fi_rows = []
    all_country_gradient_summary_rows = []
    all_stability_rows = []

    for outcome_var in outcomes:
        ml_df = build_ml_sample(
            df_input=df_input,
            outcome_var=outcome_var,
            treat_var="internet_use",
            controls=controls,
            hetero_features=hetero_features,
            sample_flag=sample_flag,
            resid_cache=resid_cache
        )

        ml_out, ate_hat, ate_sd, fi_df = fit_causal_forest_continuous(
            ml_df=ml_df,
            outcome_var=outcome_var,
            treat_var="internet_use",
            hetero_features=hetero_features,
            random_state=RANDOM_STATE
        )

        decile_df = make_decile_table(ml_out, cate_col="cate_hat")
        print_core_results(spec_name, outcome_var, ate_hat, ate_sd, fi_df, decile_df)

        #逐观测 CATE 面板
        maybe_save_csv(
            ml_out,
            SAVE_PANEL_CSV,
            os.path.join(spec_dir, f"{spec_name}_{outcome_var}_cate_panel.csv"),
            f"{spec_name}-{outcome_var} CATE 面板结果"
        )

        # feature importance
        fi_df_insert = fi_df.copy()
        fi_df_insert.insert(0, "spec_name", spec_name)
        fi_df_insert.insert(1, "spec_label", label)
        fi_df_insert.insert(2, "outcome", outcome_var)

        maybe_save_csv(
            fi_df_insert,
            SAVE_FEATURE_IMPORTANCE_CSV,
            os.path.join(spec_dir, f"{spec_name}_{outcome_var}_feature_importance.csv"),
            f"{spec_name}-{outcome_var} 特征重要性"
        )
        all_fi_rows.append(fi_df_insert)

        # CATE 十分位表
        maybe_save_csv(
            decile_df,
            SAVE_DECILE_CSV,
            os.path.join(spec_dir, f"{spec_name}_{outcome_var}_cate_decile.csv"),
            f"{spec_name}-{outcome_var} CATE 十分位表"
        )

        # country-level moderator gradient
        if RUN_COUNTRY_GRADIENT:
            for feat in hetero_features:
                country_tab, high_low_gap = make_country_level_cate_table(
                    ml_df=ml_out,
                    moderator=feat,
                    n_groups=3
                )

                print_country_gradient_result(
                    spec_name=spec_name,
                    outcome_var=outcome_var,
                    moderator=feat,
                    country_tab=country_tab,
                    high_low_gap=high_low_gap
                )

                if not country_tab.empty:
                    country_tab_to_save = country_tab.copy()
                    country_tab_to_save.insert(0, "spec_name", spec_name)
                    country_tab_to_save.insert(1, "spec_label", label)
                    country_tab_to_save.insert(2, "outcome", outcome_var)
                    country_tab_to_save.insert(3, "moderator", feat)

                    maybe_save_csv(
                        country_tab_to_save,
                        SAVE_COUNTRY_GRADIENT_CSV,
                        os.path.join(spec_dir, f"{spec_name}_{outcome_var}_{feat}_country_gradient.csv"),
                        f"{spec_name}-{outcome_var}-{feat} country-level moderator gradient"
                    )

                all_country_gradient_summary_rows.append({
                    "spec_name": spec_name,
                    "spec_label": label,
                    "outcome": outcome_var,
                    "moderator": feat,
                    "n_country": ml_out["iso3"].nunique(),
                    "group_1_cate_mean": country_tab.loc[country_tab["group"] == 1, "cate_mean"].iloc[0] if (not country_tab.empty and (country_tab["group"] == 1).any()) else np.nan,
                    "group_2_cate_mean": country_tab.loc[country_tab["group"] == 2, "cate_mean"].iloc[0] if (not country_tab.empty and (country_tab["group"] == 2).any()) else np.nan,
                    "group_3_cate_mean": country_tab.loc[country_tab["group"] == 3, "cate_mean"].iloc[0] if (not country_tab.empty and (country_tab["group"] == 3).any()) else np.nan,
                    "high_low_gap": high_low_gap
                })

        # feature-ranking stability
        need_run_stability = RUN_SEED_STABILITY and (
            (not STABILITY_ONLY_FOR_SECURE) or (outcome_var == "secure_employment")
        )

        if need_run_stability:
            stability_long_df, stability_df = run_seed_stability(
                ml_df=ml_df,
                outcome_var=outcome_var,
                treat_var="internet_use",
                hetero_features=hetero_features,
                seeds=STABILITY_SEEDS
            )

            print_stability_result(
                spec_name=spec_name,
                outcome_var=outcome_var,
                stability_df=stability_df
            )

            stability_to_save = stability_df.copy()
            stability_to_save.insert(0, "spec_name", spec_name)
            stability_to_save.insert(1, "spec_label", label)
            stability_to_save.insert(2, "outcome", outcome_var)

            maybe_save_csv(
                stability_to_save,
                SAVE_STABILITY_CSV,
                os.path.join(spec_dir, f"{spec_name}_{outcome_var}_feature_stability.csv"),
                f"{spec_name}-{outcome_var} feature-ranking stability"
            )

            stability_long_to_save = stability_long_df.copy()
            stability_long_to_save.insert(0, "spec_name", spec_name)
            stability_long_to_save.insert(1, "spec_label", label)
            stability_long_to_save.insert(2, "outcome", outcome_var)

            maybe_save_csv(
                stability_long_to_save,
                SAVE_STABILITY_LONG_CSV,
                os.path.join(spec_dir, f"{spec_name}_{outcome_var}_feature_stability_long.csv"),
                f"{spec_name}-{outcome_var} feature-ranking stability long table"
            )

            all_stability_rows.append(stability_to_save)

        #图和分箱表：只保留最需要的
        should_save_plots_for_this_spec = SAVE_PLOTS and (
            spec_name == "main_text" or SAVE_EXTENDED_PLOTS
        )

        if should_save_plots_for_this_spec:
            for feat in plot_features:
                if feat not in ml_out.columns:
                    continue

                stub = feat.replace("_base", "")
                label_x = FEATURE_LABELS.get(feat, feat)
                binned_df = make_binned_effect_table(
                    ml_df=ml_out,
                    x_col=feat,
                    cate_col="cate_hat",
                    n_bins=10
                )

                maybe_save_csv(
                    binned_df,
                    SAVE_BINNED_CSV,
                    os.path.join(spec_dir, f"{spec_name}_{outcome_var}_{stub}_binned.csv"),
                    f"{spec_name}-{outcome_var}-{stub} 分箱效应表"
                )

                plot_binned_curve(
                    tab=binned_df,
                    x_label=label_x,
                    outcome_var=outcome_var,
                    file_stub=stub,
                    output_dir=spec_dir,
                    prefix=spec_name
                )

            plot_feature_importance(
                fi_df=fi_df,
                outcome_var=outcome_var,
                output_dir=spec_dir,
                prefix=spec_name
            )

            if SAVE_CATE_HIST_PLOT:
                plot_hist_cate(
                    ml_df=ml_out,
                    outcome_var=outcome_var,
                    output_dir=spec_dir,
                    prefix=spec_name
                )

        # 总结表
        summary_rows.append({
            "spec_name": spec_name,
            "spec_label": label,
            "sample_flag": sample_flag,
            "outcome": outcome_var,
            "n_obs": len(ml_out),
            "n_country": ml_out["iso3"].nunique(),
            "n_year": ml_out["year"].nunique(),
            "mean_cate": float(ml_out["cate_hat"].mean()),
            "sd_cate": float(ml_out["cate_hat"].std(ddof=1)),
            "min_cate": float(ml_out["cate_hat"].min()),
            "p25_cate": float(ml_out["cate_hat"].quantile(0.25)),
            "p50_cate": float(ml_out["cate_hat"].quantile(0.50)),
            "p75_cate": float(ml_out["cate_hat"].quantile(0.75)),
            "max_cate": float(ml_out["cate_hat"].max()),
            "ate_like_mean": float(ate_hat),
            "ate_like_sd": float(ate_sd),
            "top_feature_1": fi_df.iloc[0]["feature"] if len(fi_df) >= 1 else "",
            "top_feature_2": fi_df.iloc[1]["feature"] if len(fi_df) >= 2 else "",
            "top_feature_3": fi_df.iloc[2]["feature"] if len(fi_df) >= 3 else ""
        })

    summary_df = pd.DataFrame(summary_rows)
    fi_all_df = pd.concat(all_fi_rows, axis=0, ignore_index=True) if all_fi_rows else pd.DataFrame()
    gradient_summary_df = pd.DataFrame(all_country_gradient_summary_rows) if all_country_gradient_summary_rows else pd.DataFrame()
    stability_all_df = pd.concat(all_stability_rows, axis=0, ignore_index=True) if all_stability_rows else pd.DataFrame()

    if not summary_df.empty:
        print("\n" + "=" * 80)
        print(f"{spec_name} 总结表")
        print("=" * 80)
        print(summary_df.to_string(index=False))

        maybe_save_csv(
            summary_df,
            SAVE_SUMMARY_CSV,
            os.path.join(spec_dir, f"{spec_name}_summary_results.csv"),
            f"{spec_name} 总结表"
        )

    if not fi_all_df.empty:
        maybe_save_csv(
            fi_all_df,
            SAVE_FEATURE_IMPORTANCE_CSV,
            os.path.join(spec_dir, f"{spec_name}_all_feature_importance.csv"),
            f"{spec_name} 全部特征重要性"
        )

    if not gradient_summary_df.empty:
        maybe_save_csv(
            gradient_summary_df,
            SAVE_COUNTRY_GRADIENT_SUMMARY_CSV,
            os.path.join(spec_dir, f"{spec_name}_country_gradient_summary.csv"),
            f"{spec_name} country-level gradient summary"
        )

    if not stability_all_df.empty:
        maybe_save_csv(
            stability_all_df,
            SAVE_STABILITY_CSV,
            os.path.join(spec_dir, f"{spec_name}_feature_stability_all.csv"),
            f"{spec_name} feature-ranking stability all"
        )

    return summary_df, fi_all_df, gradient_summary_df, stability_all_df




def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    print("\n================ 原始读取信息 ================\n")
    print("数据维度：", df.shape)
    print("列数：", len(df.columns))

    if "country_name" not in df.columns:
        df["country_name"] = df["iso3"]

    if "secure_employment" not in df.columns and "female_vulnerable_emp" in df.columns:
        df["secure_employment"] = 100 - df["female_vulnerable_emp"]
        print("已自动构造 secure_employment = 100 - female_vulnerable_emp")

    required_cols = [
        "iso3", "year", "country_name", "internet_use",
        "secure_employment", "female_vulnerable_emp",
        "log_gdp_pc_ppp", "fertility", "urbanization",
        "service_share", "trade_open", "wbl_index",
        "female_tertiary", "sample_C_complete"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少变量：{missing_cols}")

    baseline_df = make_country_baseline(
        df_input=df,
        vars_for_base=ALL_BASELINE_VARS,
        start_year=2006,
        end_year=2008,
        id_col="iso3",
        time_col="year"
    )

    safe_to_csv(
        baseline_df,
        os.path.join(OUTPUT_ROOT, "all_country_baseline_features.csv"),
        "国家基期特征总表"
    )

    df = df.merge(baseline_df, on="iso3", how="left")

    all_summary = []
    all_fi = []
    all_gradient = []
    all_stability = []
    resid_cache = {}

    for spec_name, spec_config in MODEL_SPECS.items():
        summary_df, fi_df, gradient_df, stability_df = run_one_spec(
            df_input=df,
            spec_name=spec_name,
            spec_config=spec_config,
            resid_cache=resid_cache
        )

        if not summary_df.empty:
            all_summary.append(summary_df)
        if not fi_df.empty:
            all_fi.append(fi_df)
        if not gradient_df.empty:
            all_gradient.append(gradient_df)
        if not stability_df.empty:
            all_stability.append(stability_df)

    if all_summary:
        final_summary_df = pd.concat(all_summary, axis=0, ignore_index=True)
        print("\n" + "=" * 100)
        print("双轨 ML 总结表")
        print("=" * 100)
        print(final_summary_df.to_string(index=False))

        maybe_save_csv(
            final_summary_df,
            SAVE_SUMMARY_CSV,
            os.path.join(OUTPUT_ROOT, "ml_dual_track_summary_results.csv"),
            "双轨 ML 总结表"
        )

    if all_fi:
        final_fi_df = pd.concat(all_fi, axis=0, ignore_index=True)
        maybe_save_csv(
            final_fi_df,
            SAVE_FEATURE_IMPORTANCE_CSV,
            os.path.join(OUTPUT_ROOT, "ml_dual_track_feature_importance_all.csv"),
            "双轨 ML 全部特征重要性"
        )

    if all_gradient:
        final_gradient_df = pd.concat(all_gradient, axis=0, ignore_index=True)
        maybe_save_csv(
            final_gradient_df,
            SAVE_COUNTRY_GRADIENT_SUMMARY_CSV,
            os.path.join(OUTPUT_ROOT, "ml_dual_track_country_gradient_summary.csv"),
            "双轨 ML country-level gradient summary"
        )

    if all_stability:
        final_stability_df = pd.concat(all_stability, axis=0, ignore_index=True)
        maybe_save_csv(
            final_stability_df,
            SAVE_STABILITY_CSV,
            os.path.join(OUTPUT_ROOT, "ml_dual_track_feature_stability_all.csv"),
            "双轨 ML feature-ranking stability all"
        )

    print("\n" + "=" * 100)
    print("05机器学习_增强版.py 运行结束")
    print("=" * 100)


if __name__ == "__main__":
    main()