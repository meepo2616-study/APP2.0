from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BASE_FEATURES = ["Vw", "Bave", "hd", "hb"]
PHYSICS_FEATURES = ["Fgeo_energy", "Fshape", "Fdischarge"]
EXTENDED_FEATURES = BASE_FEATURES + PHYSICS_FEATURES
G = 9.81


st.set_page_config(
    page_title="现场溃堤快速评估原型",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_prediction_model() -> tuple[Any | None, list[str], str]:
    """Load the PE-XGBoost model used by the field assessment prototype."""
    pe_path = APP_DIR / "PE_XGBoost.pkl"
    feature_path = APP_DIR / "pe_selected_features.pkl"

    if pe_path.exists():
        model = joblib.load(pe_path)
        if feature_path.exists():
            names = list(joblib.load(feature_path))
        else:
            names = list(getattr(model, "feature_names_in_", EXTENDED_FEATURES))
        n_features = int(getattr(model, "n_features_in_", len(names)))
        return model, names[:n_features], "PE-XGBoost"

    return None, EXTENDED_FEATURES, "未加载 PE-XGBoost"


def safe_divide(num: float, den: float, fallback: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-12 else fallback


def compute_physics_features(vw: float, bave: float, hd: float, hb: float) -> dict[str, float]:
    return {
        "Fgeo_energy": float(vw * hd),
        "Fshape": safe_divide(hb, bave),
        "Fdischarge": float(bave * max(hb, 0.0) ** 1.5),
        "Ip": safe_divide(vw, hd**3),
    }


def model_feature_value(feature: str, values: dict[str, float]) -> float:
    aliases = {
        "Fgeo-energy": "Fgeo_energy",
        "F_geo_energy": "Fgeo_energy",
        "Fshape": "Fshape",
        "F_shape": "Fshape",
        "Fdischarge": "Fdischarge",
        "F_discharge": "Fdischarge",
    }
    key = aliases.get(feature, feature)
    return float(values.get(key, 0.0))


def predict_qp(model: Any | None, feature_names: list[str], values: dict[str, float]) -> float | None:
    if model is None:
        return None
    row = [model_feature_value(name, values) for name in feature_names]
    # Older XGBoost versions call pandas.Int64Index when predicting DataFrames,
    # which is incompatible with pandas 2.x. A NumPy array avoids that path.
    data = np.asarray([row], dtype=float)
    return float(model.predict(data)[0])


def classify_mechanism(
    hd: float,
    b_current: float,
    hb_current: float,
    overtopping_depth: float,
    width_growth_rate: float,
    sidewall_collapse: bool,
    manual_stage: str,
) -> tuple[str, str]:
    if manual_stage != "自动判别":
        return manual_stage, "人工指定"

    depth_ratio = safe_divide(hb_current, hd)
    width_ratio = safe_divide(b_current, max(hd, 1e-6))

    if overtopping_depth <= 0.0:
        return "未漫顶或待监测阶段", "上游水位尚未形成持续漫顶条件"
    if depth_ratio < 0.15 and width_growth_rate < 0.05:
        return "初期漫顶冲刷阶段", "溃口下切较浅，局部侵蚀和入渗软化影响较明显"
    if depth_ratio < 0.55 and width_growth_rate >= 0.05:
        return "陡坎溯源与快速下切阶段", "溃口深度和宽度均处于发展过程，泄流能力快速增强"
    if sidewall_collapse or width_ratio > 2.0:
        return "侧壁失稳与横向展宽阶段", "溃口横向尺度增大，几何调整对泄流能力影响增强"
    return "溃口调整与流量衰减阶段", "溃口几何变化趋缓，流量受水量衰减和断面约束共同影响"


def stage_parameters(stage: str) -> tuple[float, float]:
    if "初期" in stage:
        return 0.42, 7.0
    if "陡坎" in stage:
        return 0.32, 10.0
    if "侧壁" in stage:
        return 0.26, 12.0
    if "衰减" in stage:
        return 0.22, 9.0
    return 0.35, 8.0


def logistic_progress(x: np.ndarray, center: float, sharpness: float) -> np.ndarray:
    raw = 1.0 / (1.0 + np.exp(-sharpness * (x - center)))
    start = raw[0]
    end = raw[-1]
    if abs(end - start) < 1e-12:
        return np.zeros_like(raw)
    return (raw - start) / (end - start)


def simulate_process(
    *,
    vw: float,
    hd: float,
    b_initial: float,
    b_final: float,
    hb_initial: float,
    hb_final: float,
    upstream_head: float,
    duration_min: float,
    dt_min: float,
    weir_coeff: float,
    stage: str,
    target_qp: float | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    duration_s = max(duration_min * 60.0, 60.0)
    dt_s = max(dt_min * 60.0, 1.0)
    t = np.arange(0.0, duration_s + dt_s, dt_s)
    x = t / duration_s
    center, sharpness = stage_parameters(stage)
    p = logistic_progress(x, center, sharpness)

    b = b_initial + (b_final - b_initial) * p
    hb = hb_initial + (hb_final - hb_initial) * p

    q_raw = np.zeros_like(t)
    cumulative = 0.0
    available_volume = max(vw, 1.0)
    h0 = max(min(upstream_head, hd), 0.05)

    for i in range(len(t)):
        depletion = max(0.0, 1.0 - cumulative / available_volume)
        effective_head = min(max(hb[i], 0.05), h0) * math.sqrt(max(depletion, 0.0))
        q_raw[i] = weir_coeff * max(b[i], 0.1) * effective_head**1.5
        cumulative += q_raw[i] * dt_s

    raw_peak = float(np.max(q_raw)) if len(q_raw) else 0.0
    if target_qp and raw_peak > 0:
        q_scaled = q_raw * (target_qp / raw_peak)
    else:
        q_scaled = q_raw.copy()

    peak_idx = int(np.argmax(q_scaled)) if len(q_scaled) else 0
    df = pd.DataFrame(
        {
            "time_min": t / 60.0,
            "breach_width_m": b,
            "breach_depth_m": hb,
            "Q_process_m3s": q_scaled,
            "Q_raw_weir_m3s": q_raw,
        }
    )
    summary = {
        "Qp_process": float(np.max(q_scaled)) if len(q_scaled) else 0.0,
        "Qp_raw_weir": raw_peak,
        "Tp_min": float(df.loc[peak_idx, "time_min"]) if len(df) else 0.0,
        "B_final": float(b[-1]) if len(b) else b_final,
        "hb_final": float(hb[-1]) if len(hb) else hb_final,
        "released_volume_raw": float(np.trapz(q_raw, t)) if len(t) > 1 else 0.0,
    }
    return df, summary


def risk_level(qp: float | None) -> tuple[str, str]:
    if qp is None:
        return "未判定", "未加载峰值流量预测模型"
    if qp <= 136:
        return "低流量等级", "可作为一般险情初判，仍需结合现场发展趋势复核"
    if qp <= 1200:
        return "中流量等级", "建议同步评估下游河道与低洼区承载能力"
    if qp <= 3000:
        return "高流量等级", "建议启动重点防护对象和下游转移预案复核"
    return "特大流量等级", "建议结合二维水动力模型开展下游淹没范围快速推演"


def format_value(value: float, unit: str = "") -> str:
    if abs(value) >= 10000:
        return f"{value:,.0f}{unit}"
    if abs(value) >= 100:
        return f"{value:,.1f}{unit}"
    return f"{value:,.3g}{unit}"


model, model_features, model_label = load_prediction_model()

st.title("现场溃堤快速评估原型")
st.caption("机制判别 - 简化过程推演 - 智能峰值预测 - 工程适用性提示")

with st.sidebar:
    st.subheader("模型状态")
    st.write(f"当前加载：{model_label}")
    st.write("模型输入：")
    st.code(", ".join(model_features), language="text")
    if model is None:
        st.error("当前目录未发现 PE_XGBoost.pkl。请先运行 train_pe_xgboost.py 或放入训练好的 PE-XGBoost 模型。")
        st.stop()

    st.subheader("基础工况")
    hd = st.number_input("堤/坝高 hd (m)", min_value=0.1, max_value=300.0, value=18.0, step=0.5)
    vw = st.number_input("溃口底部以上库容 Vw (m3)", min_value=1.0, max_value=5.0e9, value=1.3e7, step=1.0e5, format="%.0f")
    bave = st.number_input("预测/估计平均溃口宽度 Bave (m)", min_value=0.1, max_value=1000.0, value=65.0, step=1.0)
    hb_final = st.number_input("预测/估计最终溃口深度 hb (m)", min_value=0.1, max_value=300.0, value=12.0, step=0.5)
    upstream_head = st.number_input("初始有效水头 hw (m)", min_value=0.1, max_value=300.0, value=min(12.0, hd), step=0.5)

    st.subheader("现场观测")
    overtopping_depth = st.number_input("漫顶水深 (m)", min_value=0.0, max_value=20.0, value=0.4, step=0.1)
    b_current = st.number_input("当前溃口宽度 (m)", min_value=0.1, max_value=1000.0, value=8.0, step=0.5)
    hb_current = st.number_input("当前溃口深度 (m)", min_value=0.0, max_value=300.0, value=2.0, step=0.5)
    width_growth_rate = st.number_input("宽度增长率 (m/min)", min_value=0.0, max_value=100.0, value=0.08, step=0.01)
    sidewall_collapse = st.checkbox("观测到侧壁坍塌/块体失稳", value=False)
    manual_stage = st.selectbox(
        "阶段判别",
        [
            "自动判别",
            "初期漫顶冲刷阶段",
            "陡坎溯源与快速下切阶段",
            "侧壁失稳与横向展宽阶段",
            "溃口调整与流量衰减阶段",
        ],
    )

    st.subheader("过程推演")
    duration_min = st.number_input("推演时长 (min)", min_value=10.0, max_value=1440.0, value=180.0, step=10.0)
    dt_min = st.number_input("时间步长 (min)", min_value=0.1, max_value=30.0, value=2.0, step=0.5)
    weir_coeff = st.number_input("简化堰流系数", min_value=0.2, max_value=4.0, value=1.7, step=0.1)


physics = compute_physics_features(vw=vw, bave=bave, hd=hd, hb=hb_final)
all_values = {
    "Vw": vw,
    "Bave": bave,
    "hd": hd,
    "hb": hb_final,
    **physics,
}
qp_ml = predict_qp(model, model_features, all_values)
stage, stage_basis = classify_mechanism(
    hd=hd,
    b_current=b_current,
    hb_current=hb_current,
    overtopping_depth=overtopping_depth,
    width_growth_rate=width_growth_rate,
    sidewall_collapse=sidewall_collapse,
    manual_stage=manual_stage,
)
process_df, process_summary = simulate_process(
    vw=vw,
    hd=hd,
    b_initial=b_current,
    b_final=bave,
    hb_initial=max(hb_current, 0.05),
    hb_final=hb_final,
    upstream_head=upstream_head,
    duration_min=duration_min,
    dt_min=dt_min,
    weir_coeff=weir_coeff,
    stage=stage,
    target_qp=qp_ml,
)
level, level_note = risk_level(qp_ml)

top_left, top_mid, top_right = st.columns(3)
top_left.metric("智能预测峰值流量 Qp", "未加载" if qp_ml is None else f"{qp_ml:,.2f} m3/s")
top_mid.metric("过程线峰现时间 Tp", f"{process_summary['Tp_min']:.1f} min")
top_right.metric("风险等级", level)

tabs = st.tabs(["综合结果", "过程线", "关键参数组合", "工程适用性"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("机制判别")
        st.write(f"判别结果：**{stage}**")
        st.write(f"判别依据：{stage_basis}")
        st.write(f"工程提示：{level_note}")

        raw_peak = process_summary["Qp_raw_weir"]
        if qp_ml is not None and raw_peak > 0:
            diff = abs(qp_ml - raw_peak) / max(qp_ml, 1.0)
            st.write(f"智能峰值与未缩放简化堰流峰值差异：{diff * 100:.1f}%")
            if diff > 0.3:
                st.warning("两类估计差异较大，建议复核溃口宽度、水头和库容输入，必要时接入二维水动力模型。")

    with right:
        st.subheader("峰值与过程摘要")
        summary_df = pd.DataFrame(
            [
                ["智能预测峰值 Qp", np.nan if qp_ml is None else qp_ml, "m3/s"],
                ["简化堰流原始峰值", process_summary["Qp_raw_weir"], "m3/s"],
                ["过程线峰现时间 Tp", process_summary["Tp_min"], "min"],
                ["最终溃口宽度", process_summary["B_final"], "m"],
                ["最终溃口深度", process_summary["hb_final"], "m"],
                ["相对蓄水强度 Ip", physics["Ip"], "-"],
            ],
            columns=["指标", "数值", "单位"],
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("流量过程线")
    st.line_chart(process_df.set_index("time_min")[["Q_process_m3s", "Q_raw_weir_m3s"]])

    st.subheader("溃口几何演化")
    st.line_chart(process_df.set_index("time_min")[["breach_width_m", "breach_depth_m"]])

with tabs[2]:
    st.subheader("输入变量与物理交互因子")
    factor_rows = [
        ["Vw", vw, "m3", "溃口底部以上库容"],
        ["Bave", bave, "m", "平均溃口宽度"],
        ["hd", hd, "m", "堤/坝高"],
        ["hb", hb_final, "m", "溃口深度"],
        ["Fgeo_energy", physics["Fgeo_energy"], "m4", "水量-水头组合尺度"],
        ["Fshape", physics["Fshape"], "-", "溃口深宽比例"],
        ["Fdischarge", physics["Fdischarge"], "m2.5", "断面泄流能力代理量"],
        ["Ip", physics["Ip"], "-", "相对蓄水强度"],
    ]
    factors_df = pd.DataFrame(factor_rows, columns=["参数", "数值", "单位", "含义"])
    st.dataframe(factors_df, hide_index=True, use_container_width=True)

    st.subheader("模型输入向量")
    model_input_df = pd.DataFrame(
        [[name, model_feature_value(name, all_values)] for name in model_features],
        columns=["模型特征", "输入值"],
    )
    st.dataframe(model_input_df, hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("适用性说明")
    st.write(
        "该原型用于现场资料有限条件下的峰值流量和过程线快速初判。"
        "当前过程线为简化堰流关系和几何增长函数生成，并按智能预测峰值进行缩放；"
        "其作用是提供应急估算边界条件，不能替代完整二维水动力演算。"
    )
    st.write(
        "若需要下游淹没范围、到达时间、最大水深和完整洪水演进过程，"
        "建议将本原型输出的 Q(t) 作为边界条件接入 MIKE21 等二维水动力模型。"
    )

    report_df = process_df.copy()
    report_df["Qp_ml_m3s"] = np.nan if qp_ml is None else qp_ml
    report_df["mechanism_stage"] = stage
    report_df["risk_level"] = level
    csv = report_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载过程线与评估结果 CSV",
        data=csv,
        file_name="field_breach_assessment_result.csv",
        mime="text/csv",
    )
