"""Streamlit dashboard for the Knowledge Distillation Lab.

Run:  streamlit run app_distill.py
Reads: distill_results.csv (from distill_lab.py) and benchmark_results.csv (from benchmark.py)
"""
import os

import pandas as pd
import streamlit as st

RESULTS = "distill_results.csv"
BENCH = "benchmark_results.csv"

st.set_page_config(page_title="Knowledge Distillation Lab", layout="wide")
st.title("Knowledge Distillation Lab")
st.caption("BERT-base teacher to a smaller student | 6-class emotion classification")

if not os.path.exists(RESULTS):
    st.warning("`distill_results.csv` not found. Run `distill_lab.py` first.")
    st.stop()

df = pd.read_csv(RESULTS).drop_duplicates(subset="run", keep="last")
df["train_limit"] = df["train_limit"].astype(str)

if os.path.exists(BENCH):
    bench = pd.read_csv(BENCH)
else:
    bench = pd.DataFrame(columns=["run", "device", "lat_ms_mean", "lat_ms_p95", "throughput_sps"])

teachers = df[df["mode"] == "teacher"]
students = sorted(df.loc[df["mode"] != "teacher", "model"].unique())
if teachers.empty or not students:
    st.info("Need at least one teacher run and one student run to compare.")
    st.dataframe(df)
    st.stop()

# ------------------------------------------------------------ sidebar
student = st.sidebar.selectbox("Student model", students)
sizes = sorted(df.loc[(df["mode"] != "teacher") & (df["model"] == student), "train_limit"].unique())
size = st.sidebar.selectbox("Training data used by student", sizes)
devices = sorted(bench["device"].dropna().unique())
device = st.sidebar.selectbox("Benchmark device", devices) if devices else None
st.sidebar.caption("Best T / alpha is chosen on the VALIDATION set. Test accuracy is reported only.")

# ------------------------------------------------------------ helpers
NUM = ["params_M", "size_mb", "test_acc", "test_f1", "train_time_s", "lat_ms_mean", "throughput_sps"]


def attach_bench(rows):
    out = rows.copy()
    if device is None:
        out["lat_ms_mean"] = float("nan")
        out["throughput_sps"] = float("nan")
        return out
    b = bench[bench["device"] == device][["run", "lat_ms_mean", "throughput_sps"]]
    return out.merge(b, on="run", how="left")


def best_distill(d):
    """Rows of the (T, alpha) config with the best mean VALIDATION accuracy."""
    if d.empty:
        return d
    g = d.groupby(["temperature", "alpha"])["best_val_acc"].mean()
    t, a = g.idxmax()
    return d[(d["temperature"] == t) & (d["alpha"] == a)]


def summary(rows, label):
    rows = attach_bench(rows)
    out = {"model": label, "seeds": len(rows)}
    for c in NUM:
        out[c] = rows[c].mean()
    return out


sub = df[(df["model"] == student) & (df["train_limit"] == size)]
plain = sub[sub["mode"] == "student_plain"]
distill_all = sub[sub["mode"] == "student_distill"]
best = best_distill(distill_all)

t_sum = summary(teachers, "Teacher")
p_sum = summary(plain, "Plain student (labels only)") if not plain.empty else None
d_sum = None
if not best.empty:
    bt, ba = best.iloc[0]["temperature"], best.iloc[0]["alpha"]
    d_sum = summary(best, f"Distilled student (T={bt:g}, alpha={ba:g})")

# ------------------------------------------------------------ headline
st.header("1. Teacher vs plain student vs distilled student")
sums = [s for s in (t_sum, p_sum, d_sum) if s is not None]
comp = pd.DataFrame(sums).set_index("model")
show = comp.copy()
show["test_acc"] = show["test_acc"] * 100
show["test_f1"] = show["test_f1"] * 100
show = show.rename(columns={"test_acc": "test_acc_%", "test_f1": "test_f1_%"})
st.dataframe(show.round(2), use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
if p_sum and d_sum:
    c1.metric("Distilled vs plain (acc, points)",
              f"{(d_sum['test_acc'] - p_sum['test_acc']) * 100:+.2f}")
if d_sum:
    c2.metric("Teacher minus distilled (acc, points)",
              f"{(t_sum['test_acc'] - d_sum['test_acc']) * 100:.2f}")
    c3.metric("Parameter reduction vs teacher",
              f"{(1 - d_sum['params_M'] / t_sum['params_M']) * 100:.0f}%")
    if device and pd.notna(d_sum["lat_ms_mean"]) and pd.notna(t_sum["lat_ms_mean"]):
        c4.metric(f"Latency speedup ({device})",
                  f"{t_sum['lat_ms_mean'] / d_sum['lat_ms_mean']:.2f}x")
if p_sum is None or d_sum is None:
    st.warning("Run both `student_plain` and `student_distill` for this student/data setting.")

if device and comp[["lat_ms_mean"]].notna().any().any():
    b1, b2 = st.columns(2)
    b1.subheader(f"Latency, batch size 1 ({device}, ms)")
    b1.bar_chart(comp["lat_ms_mean"])
    b2.subheader(f"Throughput, batch 32 ({device}, samples/s)")
    b2.bar_chart(comp["throughput_sps"])
else:
    st.info("No benchmark data yet. Run `benchmark.py` to add latency and throughput.")

# ------------------------------------------------------------ ablation
st.header("2. Temperature and alpha ablation (validation accuracy)")
if distill_all.empty:
    st.info("No distillation runs for this setting yet.")
else:
    pv = distill_all.pivot_table(index="alpha", columns="temperature",
                                 values="best_val_acc", aggfunc="mean")
    if not plain.empty:
        st.write(f"Plain student validation accuracy (reference): "
                 f"**{plain['best_val_acc'].mean():.4f}**")
    st.dataframe(pv.round(4), use_container_width=True)
    st.line_chart(pv)
    st.caption("Rows = alpha (weight of label loss), columns = temperature T.")

# ------------------------------------------------------------ data efficiency
st.header("3. Does distillation help more with less data?")
eff = []
for s in sizes:
    s_sub = df[(df["model"] == student) & (df["train_limit"] == s)]
    p = s_sub[s_sub["mode"] == "student_plain"]
    d = best_distill(s_sub[s_sub["mode"] == "student_distill"])
    if len(p) and len(d):
        eff.append({"train_size": s,
                    "plain_test_acc_%": p["test_acc"].mean() * 100,
                    "distill_test_acc_%": d["test_acc"].mean() * 100,
                    "gain_points": (d["test_acc"].mean() - p["test_acc"].mean()) * 100})
if len(eff) >= 2:
    eff_df = pd.DataFrame(eff).set_index("train_size")
    st.dataframe(eff_df.round(2), use_container_width=True)
    st.bar_chart(eff_df[["plain_test_acc_%", "distill_test_acc_%"]])
else:
    st.info("Run the student with different `--train_limit` values (e.g. 500, 1000, full) "
            "to compare data regimes.")

with st.expander("All runs (raw)"):
    st.dataframe(df, use_container_width=True)
