from __future__ import annotations

# 论文附录复现代码。
# 问题三：计算扰流特征重要性、PCA 与聚类结果。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_evolution_ridge_ann import OUT_DIR, loadData
from physical_feature_library import enhancedFeatureFrame
from compact_rbf_krr import columnsForFeatureSet


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "verification_figures"
REPORT = Path(__file__).resolve().parent / "verification_reports" / "DISTURBANCE_MECHANISM_ANALYSIS.md"
FAILURE_MODE_FILE = ROOT / "code" / "outputs" / "krr_leave_one_group_failure_modes.csv"

RANDOM_SEED = 20260711
CLUSTER_K_RANGE = range(2, 6)


def registerPlotFont() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simsun.ttc")]:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("MechanismCN", str(font_path), subfontIndex=0))
            return "MechanismCN"
        except Exception:
            pass
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def ensureDirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ScaleParams:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "ScaleParams":
        arr = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(columns, mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        arr = frame[self.columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        return (arr - self.mean) / self.scale


def compactFeatureFrame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = enhancedFeatureFrame(data)
    cols = columnsForFeatureSet(features, "compact_physical")
    return features, cols


def rankAverage(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def aucBinary(feature: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankAverage(feature)
    rank_pos = float(ranks[y == 1].sum())
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def logLoss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fitLogisticL2(x: np.ndarray, y: np.ndarray, l2: float = 0.05, epochs: int = 8000, lr: float = 0.03) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    beta = rng.normal(0.0, 0.02, size=x.shape[1])
    intercept = float(math.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))
    for _ in range(epochs):
        z = x @ beta + intercept
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        err = p - y
        grad_beta = x.T @ err / len(y) + 2.0 * l2 * beta
        grad_intercept = float(err.mean())
        beta -= lr * grad_beta
        intercept -= lr * grad_intercept
    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta + intercept, -40, 40)))
    return beta, intercept, p


def permutationImportanceLogistic(x: np.ndarray, y: np.ndarray, beta: np.ndarray, intercept: float, repeats: int = 30) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED + 19)
    base_p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta + intercept, -40, 40)))
    base_loss = logLoss(y, base_p)
    importances = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        losses = []
        for _ in range(repeats):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            p = 1.0 / (1.0 + np.exp(-np.clip(xp @ beta + intercept, -40, 40)))
            losses.append(logLoss(y, p) - base_loss)
        importances[j] = float(np.mean(losses))
    return importances


def featureImportance(data: pd.DataFrame, features: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = (~data["disturbance_id"].eq("D0")).astype(int).to_numpy()
    scaler = ScaleParams.fit(features, cols)
    x = scaler.transform(features)
    beta, intercept, _ = fitLogisticL2(x, y)
    perm = permutationImportanceLogistic(x, y, beta, intercept)

    rows = []
    for j, col in enumerate(cols):
        values = features[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        d0 = values[y == 0]
        disturbed = values[y == 1]
        n0, n1 = len(d0), len(disturbed)
        mean0 = float(d0.mean())
        mean1 = float(disturbed.mean())
        sd0 = float(d0.std(ddof=1)) if n0 > 1 else 0.0
        sd1 = float(disturbed.std(ddof=1)) if n1 > 1 else 0.0
        pooled = math.sqrt(((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / max(n0 + n1 - 2, 1))
        effect = abs(mean1 - mean0) / max(pooled, 1e-12)
        auc = aucBinary(values, y)
        separation_auc = max(auc, 1.0 - auc) if pd.notna(auc) else float("nan")
        rows.append(
            {
                "feature": col,
                "mean_D0": mean0,
                "mean_disturbed": mean1,
                "effect_size_abs": effect,
                "auc_disturbed_gt_D0": auc,
                "separation_auc": separation_auc,
                "logistic_coef": float(beta[j]),
                "logistic_abs_coef": abs(float(beta[j])),
                "permutation_logloss_increase": float(perm[j]),
            }
        )
    out = pd.DataFrame(rows)
    for metric in ["effect_size_abs", "separation_auc", "logistic_abs_coef", "permutation_logloss_increase"]:
        out[f"rank_{metric}"] = out[metric].rank(ascending=False, method="average")
    out["composite_rank"] = out[
        [
            "rank_effect_size_abs",
            "rank_separation_auc",
            "rank_logistic_abs_coef",
            "rank_permutation_logloss_increase",
        ]
    ].mean(axis=1)
    return out.sort_values(["composite_rank", "effect_size_abs"], ascending=[True, False]).reset_index(drop=True)


def pcaSvd(x: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    eig = (s**2) / max(len(x) - 1, 1)
    explained = eig / max(eig.sum(), 1e-12)
    return coords, explained[:n_components], vt[:n_components]


def pairwiseDistances(x: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1)[:, None]
    d2 = np.maximum(x2 + x2.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(d2)


def kmeansLabels(x: np.ndarray, k: int, n_init: int = 20, max_iter: int = 200) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED + k)
    best_labels = None
    best_inertia = float("inf")
    n = len(x)
    for _ in range(n_init):
        centroids = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dist = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                if np.any(labels == c):
                    centroids[c] = x[labels == c].mean(axis=0)
                else:
                    centroids[c] = x[rng.integers(0, n)]
        inertia = float(np.sum((x - centroids[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels if best_labels is not None else np.zeros(n, dtype=int), best_inertia


def wardLabels(x: np.ndarray, k: int) -> np.ndarray:
    clusters = {i: [i] for i in range(len(x))}
    means = {i: x[i].copy() for i in range(len(x))}
    sizes = {i: 1 for i in range(len(x))}
    next_id = len(x)
    while len(clusters) > k:
        keys = list(clusters)
        best_pair = None
        best_delta = float("inf")
        for a_pos in range(len(keys)):
            a = keys[a_pos]
            for b in keys[a_pos + 1 :]:
                delta = sizes[a] * sizes[b] / (sizes[a] + sizes[b]) * float(np.sum((means[a] - means[b]) ** 2))
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        merged = clusters[a] + clusters[b]
        clusters[next_id] = merged
        sizes[next_id] = sizes[a] + sizes[b]
        means[next_id] = (sizes[a] * means[a] + sizes[b] * means[b]) / sizes[next_id]
        for key in [a, b]:
            del clusters[key]
            del means[key]
            del sizes[key]
        next_id += 1
    labels = np.zeros(len(x), dtype=int)
    for c, members in enumerate(clusters.values()):
        labels[members] = c
    return labels


def gmmDiagLabels(x: np.ndarray, k: int, n_init: int = 10, max_iter: int = 150) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED + 100 + k)
    n, p = x.shape
    best_labels = None
    best_ll = -float("inf")
    for _ in range(n_init):
        labels, _ = kmeansLabels(x, k, n_init=1, max_iter=50)
        means = np.vstack([x[labels == c].mean(axis=0) if np.any(labels == c) else x[rng.integers(0, n)] for c in range(k)])
        vars_ = np.vstack(
            [x[labels == c].var(axis=0) + 1e-4 if np.any(labels == c) else np.ones(p) for c in range(k)]
        )
        weights = np.array([max(np.mean(labels == c), 1.0 / n) for c in range(k)])
        weights = weights / weights.sum()
        prev_ll = -float("inf")
        resp = np.zeros((n, k))
        for _ in range(max_iter):
            log_prob = np.zeros((n, k))
            for c in range(k):
                log_det = np.sum(np.log(vars_[c]))
                quad = np.sum((x - means[c]) ** 2 / vars_[c], axis=1)
                log_prob[:, c] = math.log(weights[c] + 1e-12) - 0.5 * (p * math.log(2.0 * math.pi) + log_det + quad)
            max_log = log_prob.max(axis=1, keepdims=True)
            prob = np.exp(log_prob - max_log)
            denom = np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
            resp = prob / denom
            ll = float(np.sum(max_log[:, 0] + np.log(denom[:, 0])))
            nk = np.maximum(resp.sum(axis=0), 1e-8)
            weights = nk / n
            means = (resp.T @ x) / nk[:, None]
            for c in range(k):
                diff = x - means[c]
                vars_[c] = (resp[:, c][:, None] * diff * diff).sum(axis=0) / nk[c] + 1e-4
            if abs(ll - prev_ll) < 1e-5:
                break
            prev_ll = ll
        labels = resp.argmax(axis=1)
        if prev_ll > best_ll and len(np.unique(labels)) == k:
            best_ll = prev_ll
            best_labels = labels.copy()
    return best_labels if best_labels is not None else np.zeros(n, dtype=int), best_ll


def clusterMetrics(x: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    k = len(unique)
    n = len(x)
    if k <= 1 or k >= n:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"), "davies_bouldin": float("nan")}

    dist = pairwiseDistances(x)
    sil_values = []
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            a = 0.0
        else:
            a = float(dist[i, same].sum() / (same.sum() - 1))
        b = float("inf")
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            b = min(b, float(dist[i, other].mean()))
        sil_values.append((b - a) / max(a, b, 1e-12))
    silhouette = float(np.mean(sil_values))

    overall = x.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0
    centroids = []
    scatters = []
    for c in unique:
        xc = x[labels == c]
        centroid = xc.mean(axis=0)
        centroids.append(centroid)
        ss_between += len(xc) * float(np.sum((centroid - overall) ** 2))
        within = float(np.sum((xc - centroid) ** 2))
        ss_within += within
        scatters.append(float(np.sqrt(within / max(len(xc), 1))))
    ch = (ss_between / (k - 1)) / max(ss_within / (n - k), 1e-12)

    db_values = []
    for i in range(k):
        worst = -float("inf")
        for j in range(k):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            worst = max(worst, (scatters[i] + scatters[j]) / max(d, 1e-12))
        db_values.append(worst)
    db = float(np.mean(db_values))
    return {"silhouette": silhouette, "calinski_harabasz": float(ch), "davies_bouldin": db}


def relabelBySize(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    order = [old for old, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels], dtype=int)


def runClustering(x: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    labels_by_model: dict[str, np.ndarray] = {}
    for k in CLUSTER_K_RANGE:
        for method in ["kmeans", "ward", "gmm_diag"]:
            if method == "kmeans":
                labels, inertia = kmeansLabels(x, k)
                extra = {"inertia": inertia, "log_likelihood": float("nan")}
            elif method == "ward":
                labels = wardLabels(x, k)
                extra = {"inertia": float(np.sum((x - np.vstack([x[labels == c].mean(axis=0) for c in np.unique(labels)])[labels]) ** 2)), "log_likelihood": float("nan")}
            else:
                labels, ll = gmmDiagLabels(x, k)
                extra = {"inertia": float("nan"), "log_likelihood": ll}
            labels = relabelBySize(labels)
            metrics = clusterMetrics(x, labels)
            model_id = f"{method}_k{k}"
            labels_by_model[model_id] = labels
            rows.append({"cluster_model": model_id, "method": method, "k": k, **metrics, **extra})
    quality = pd.DataFrame(rows)
    quality["rank_silhouette"] = quality["silhouette"].rank(ascending=False, method="average")
    quality["rank_db"] = quality["davies_bouldin"].rank(ascending=True, method="average")
    quality["rank_ch"] = quality["calinski_harabasz"].rank(ascending=False, method="average")
    quality["overall_rank"] = quality[["rank_silhouette", "rank_db", "rank_ch"]].mean(axis=1)
    quality = quality.sort_values(["overall_rank", "rank_silhouette"]).reset_index(drop=True)
    return quality, labels_by_model


def majority(values: pd.Series) -> str:
    counts = Counter(str(v) for v in values.dropna().tolist())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def clusterCrossTables(
    disturbed: pd.DataFrame,
    labels: np.ndarray,
    failure_modes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = disturbed[["window_id", "date", "flow_point", "disturbance_id"]].copy()
    windows["cluster"] = labels

    cross_dist = pd.crosstab(windows["cluster"], windows["disturbance_id"]).reset_index()
    cross_dist.columns = [str(c) for c in cross_dist.columns]

    group_cluster = (
        windows.groupby(["date", "flow_point", "disturbance_id"], sort=True)["cluster"]
        .agg(lambda s: int(Counter(s.tolist()).most_common(1)[0][0]))
        .reset_index()
    )
    merged = group_cluster.merge(
        failure_modes,
        on=["date", "flow_point", "disturbance_id"],
        how="left",
        validate="one_to_one",
    )
    merged["mechanism_note"] = merged["mechanism_note"].fillna("unlabeled_small_group")
    merged["threshold_failure_type"] = merged["threshold_failure_type"].fillna("unlabeled_small_group")
    cross_mech = pd.crosstab(merged["cluster"], merged["mechanism_note"]).reset_index()
    cross_mech.columns = [str(c) for c in cross_mech.columns]
    cross_fail = pd.crosstab(merged["cluster"], merged["threshold_failure_type"]).reset_index()
    cross_fail.columns = [str(c) for c in cross_fail.columns]
    summary = (
        merged.groupby("cluster", sort=True)
        .agg(
            group_count=("disturbance_id", "size"),
            mean_abs_error_pct=("abs_mean_error_pct", "mean"),
            mean_sd_pct=("repeatability_sd_pct", "mean"),
            mean_ood=("ood_nearest_euclidean_mean", "mean"),
            dominant_mechanism=("mechanism_note", majority),
        )
        .reset_index()
    )
    return cross_dist, cross_mech, cross_fail, summary


def drawFeatureImportancePdf(df: pd.DataFrame) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError:
        return
    path = FIG_DIR / "fig_disturbance_feature_importance.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    page_w, page_h = landscape(A4)
    font = registerPlotFont()
    label_map = {
        "profile_swirl": "旋流代理",
        "profile_ab_abs": "A/B 方向不对称",
        "ab_abs_mean": "A/B 绝对差均值",
        "lagrange_volume_m3_minus_phys6_frac": "Lagrange 与 Phys6 相对差",
        "chord_edge_minus_inner_frac": "边缘与内侧声道相对差",
        "profile_edge_inner": "边缘与内侧剖面差",
        "owics_volume_m3_minus_phys6_frac": "OWICS 与 Phys6 相对差",
        "zero_rate_med": "零点速率中位数",
    }
    top = df.head(8).iloc[::-1]
    x0, y0 = 225, 62
    w, h = page_w - 305, page_h - 125
    vals = top["effect_size_abs"].to_numpy(dtype=float)
    vmax = max(1e-6, float(vals.max()) * 1.15)
    bar_h = h / len(top) * 0.72
    c.setFont(font, 16)
    for i, (_, row) in enumerate(top.iterrows()):
        y = y0 + i * h / len(top)
        bw = w * float(row["effect_size_abs"]) / vmax
        c.setFillColor(colors.HexColor("#1F5AA6"))
        c.rect(x0, y, bw, bar_h, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawRightString(x0 - 8, y + 3, label_map.get(str(row["feature"]), str(row["feature"])))
        c.drawString(x0 + bw + 6, y + 3, f"{row['effect_size_abs']:.2f}")
    c.setFont(font, 17)
    c.drawCentredString(x0 + w / 2, 28, "标准化效应量绝对值")
    c.save()


def drawPcaPdf(coords: pd.DataFrame, explained: pd.DataFrame, cluster_col: str = "best_cluster") -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError:
        return
    path = FIG_DIR / "fig_disturbance_pca_clusters.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    page_w, page_h = landscape(A4)
    font = registerPlotFont()
    x0, y0, w, h = 70, 60, page_w - 150, page_h - 120
    x = coords["PC1"].to_numpy(dtype=float)
    y = coords["PC2"].to_numpy(dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    pad_x = max((xmax - xmin) * 0.08, 1e-6)
    pad_y = max((ymax - ymin) * 0.08, 1e-6)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    palette = [
        colors.HexColor("#4C78A8"),
        colors.HexColor("#F58518"),
        colors.HexColor("#54A24B"),
        colors.HexColor("#B279A2"),
        colors.HexColor("#E45756"),
        colors.HexColor("#72B7B2"),
    ]
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.3)
    c.line(x0, y0, x0 + w, y0)
    c.line(x0, y0, x0, y0 + h)
    c.setFont(font, 16)
    for tick in range(5):
        tx = xmin + (xmax - xmin) * tick / 4
        ty = ymin + (ymax - ymin) * tick / 4
        px = x0 + w * tick / 4
        py = y0 + h * tick / 4
        c.setStrokeColor(colors.HexColor("#D0D0D0"))
        c.setLineWidth(0.6)
        c.line(px, y0, px, y0 + h)
        c.line(x0, py, x0 + w, py)
        c.setFillColor(colors.black)
        c.drawCentredString(px, y0 - 17, f"{tx:.1f}")
        c.drawRightString(x0 - 8, py - 3, f"{ty:.1f}")
    for _, row in coords.iterrows():
        px = x0 + w * (float(row["PC1"]) - xmin) / (xmax - xmin)
        py = y0 + h * (float(row["PC2"]) - ymin) / (ymax - ymin)
        cluster = int(row[cluster_col])
        c.setFillColor(palette[cluster % len(palette)])
        c.circle(px, py, 4.0, fill=1, stroke=0)
    pc1 = float(explained.loc[explained["component"].eq("PC1"), "explained_variance_ratio"].iloc[0])
    pc2 = float(explained.loc[explained["component"].eq("PC2"), "explained_variance_ratio"].iloc[0])
    c.setFillColor(colors.black)
    c.setFont(font, 16)
    c.drawCentredString(x0 + w / 2, 28, f"PC1 ({pc1:.1%})")
    c.saveState()
    c.translate(26, y0 + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, f"PC2 ({pc2:.1%})")
    c.restoreState()
    legend_x, legend_y = x0 + w - 165, y0 + h - 18
    for cluster, label in [(0, "簇 0：其余扰流为主"), (1, "簇 1：D3 为主")]:
        c.setFillColor(palette[cluster])
        c.circle(legend_x, legend_y - cluster * 20, 4.5, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont(font, 16)
        c.drawString(legend_x + 11, legend_y - 4 - cluster * 20, label)
    c.save()


def markdownTable(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_无数据。_"
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    use = use.astype(str)
    headers = list(use.columns)
    rows = use.values.tolist()
    widths = []
    for i, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[i]) for row in rows)))
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def writeReport(
    feature_rank: pd.DataFrame,
    explained: pd.DataFrame,
    quality: pd.DataFrame,
    cross_dist: pd.DataFrame,
    cross_mech: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    best_model_id: str,
) -> None:
    feature_cols = [
        "feature",
        "effect_size_abs",
        "separation_auc",
        "logistic_abs_coef",
        "permutation_logloss_increase",
        "composite_rank",
    ]
    quality_cols = ["cluster_model", "method", "k", "silhouette", "calinski_harabasz", "davies_bouldin", "overall_rank"]
    best_quality = quality.loc[quality["cluster_model"].eq(best_model_id)].iloc[0]
    best_silhouette = float(best_quality["silhouette"])
    decision = (
        "当前最佳聚类为 K=2，Silhouette 约为 "
        f"`{best_silhouette:.3f}`。聚类与扰流编号交叉表显示，第二簇主要对应 D3，而 D1、D2、D4、D5、D6、D7、D8 "
        "仍集中在同一大簇中；聚类与失效机制交叉表也显示，系统漂移型和重复性失稳型没有被清晰分离。因此当前证据不足以支持把硬聚类或软门控专家升级为最终主模型。"
        "更稳妥的结论是：D1-D8 在 compact physical 特征空间中存在局部结构，但总体仍呈连续重叠分布，统一 RBF-KRR 连续相似性补偿仍是主线；聚类只作为解释性分支和后续定向增强候选。"
    )
    text = f"""# 问题3：扰流特征重要性、PCA 聚类与失效机制对齐

## 1. 分析目标

当前最终主模型已冻结为 `krr_phys6_volume_m3_compact_physical_g0.1_lam0.1_none`。本报告不训练新的最终模型，而是服务于问题3：解释 D1-D8 扰流在在线特征空间中的结构，并判断扰流聚类是否足够清晰，是否有资格作为后续软门控增强分支。

分析对象仅包括 D1-D8 扰流窗口；`20260607-60-D0` 属于正常锚点缺失型 OOD，不进入扰流聚类解释。

## 2. 扰流特征重要性

特征重要性从三类证据综合得到：D0 与扰流窗口的标准化效应量、单特征 AUC、L2 Logistic 标准化系数和 permutation logloss increase。

{markdownTable(feature_rank[feature_cols], 15)}

这些特征的理论含义与论文库一致：剖面类特征刻画速度剖面的中心-边缘和上下差异，A/B 类特征刻画声路方向不对称，动态/零点类特征刻画测量稳定性。

## 3. PCA 结果

PCA 仅在 D1-D8 的 compact physical 特征上进行。

{markdownTable(explained, 5)}

## 4. 聚类质量比较

比较 KMeans、Ward 和对角 GMM，聚类数 `K=2--5`，评价指标包括 Silhouette、Calinski-Harabasz 和 Davies-Bouldin。当前综合排序第一的聚类模型为：

```text
{best_model_id}
```

{markdownTable(quality[quality_cols], 12)}

## 5. 聚类与扰流编号交叉表

{markdownTable(cross_dist, 20)}

## 6. 聚类与失效机制交叉表

{markdownTable(cross_mech, 20)}

## 7. 聚类组统计

{markdownTable(cluster_summary, 20)}

## 8. 结论

{decision}

## 9. 输出文件

- `code/disturbance_mechanism_analysis.py`
- `code/outputs/disturbance_feature_importance.csv`
- `code/outputs/disturbance_pca_coordinates.csv`
- `code/outputs/disturbance_pca_explained_variance.csv`
- `code/outputs/disturbance_cluster_quality.csv`
- `code/outputs/disturbance_cluster_labels.csv`
- `code/outputs/disturbance_cluster_cross_disturbance.csv`
- `code/outputs/disturbance_cluster_cross_mechanism.csv`
- `code/outputs/disturbance_cluster_cross_failure_type.csv`
- `code/outputs/disturbance_cluster_summary.csv`
- `figures/fig_disturbance_feature_importance.pdf`
- `figures/fig_disturbance_pca_clusters.pdf`
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    ensureDirs()
    data = loadData()
    features, cols = compactFeatureFrame(data)
    feature_rank = featureImportance(data, features, cols)
    feature_rank.to_csv(OUT_DIR / "disturbance_feature_importance.csv", index=False, encoding="utf-8-sig")

    disturbed = data.loc[~data["disturbance_id"].eq("D0")].reset_index(drop=True)
    disturbed_features = enhancedFeatureFrame(disturbed)
    disturbed_cols = columnsForFeatureSet(disturbed_features, "compact_physical")
    scaler = ScaleParams.fit(disturbed_features, disturbed_cols)
    x = scaler.transform(disturbed_features)
    coords, explained_ratio, loadings = pcaSvd(x, n_components=5)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(explained_ratio))],
            "explained_variance_ratio": explained_ratio,
            "cumulative_ratio": np.cumsum(explained_ratio),
        }
    )
    pca_coords = disturbed[["window_id", "date", "flow_point", "disturbance_id"]].copy()
    for i in range(coords.shape[1]):
        pca_coords[f"PC{i+1}"] = coords[:, i]
    loading_rows = []
    for i in range(loadings.shape[0]):
        for j, col in enumerate(disturbed_cols):
            loading_rows.append({"component": f"PC{i+1}", "feature": col, "loading": float(loadings[i, j])})
    loadings_df = pd.DataFrame(loading_rows)

    quality, labels_by_model = runClustering(x)
    best_model_id = str(quality.iloc[0]["cluster_model"])
    best_labels = labels_by_model[best_model_id]
    pca_coords["best_cluster"] = best_labels
    labels_out = disturbed[["window_id", "date", "flow_point", "disturbance_id"]].copy()
    for model_id, labels in labels_by_model.items():
        labels_out[model_id] = labels
    labels_out["best_cluster_model"] = best_model_id
    labels_out["best_cluster"] = best_labels

    if FAILURE_MODE_FILE.exists():
        failure_modes = pd.read_csv(FAILURE_MODE_FILE, encoding="utf-8-sig")
        failure_modes["date"] = failure_modes["date"].astype(str)
        disturbed["date"] = disturbed["date"].astype(str)
    else:
        failure_modes = pd.DataFrame(columns=["date", "flow_point", "disturbance_id", "mechanism_note", "threshold_failure_type"])

    cross_dist, cross_mech, cross_fail, cluster_summary = clusterCrossTables(disturbed, best_labels, failure_modes)

    explained.to_csv(OUT_DIR / "disturbance_pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    pca_coords.to_csv(OUT_DIR / "disturbance_pca_coordinates.csv", index=False, encoding="utf-8-sig")
    loadings_df.to_csv(OUT_DIR / "disturbance_pca_loadings.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT_DIR / "disturbance_cluster_quality.csv", index=False, encoding="utf-8-sig")
    labels_out.to_csv(OUT_DIR / "disturbance_cluster_labels.csv", index=False, encoding="utf-8-sig")
    cross_dist.to_csv(OUT_DIR / "disturbance_cluster_cross_disturbance.csv", index=False, encoding="utf-8-sig")
    cross_mech.to_csv(OUT_DIR / "disturbance_cluster_cross_mechanism.csv", index=False, encoding="utf-8-sig")
    cross_fail.to_csv(OUT_DIR / "disturbance_cluster_cross_failure_type.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(OUT_DIR / "disturbance_cluster_summary.csv", index=False, encoding="utf-8-sig")

    drawFeatureImportancePdf(feature_rank)
    drawPcaPdf(pca_coords, explained)
    writeReport(feature_rank, explained, quality, cross_dist, cross_mech, cluster_summary, best_model_id)

    print("Top feature importance:")
    print(feature_rank[["feature", "effect_size_abs", "separation_auc", "logistic_abs_coef", "permutation_logloss_increase"]].head(10).to_string(index=False))
    print("Cluster quality:")
    print(quality[["cluster_model", "silhouette", "calinski_harabasz", "davies_bouldin", "overall_rank"]].head(8).to_string(index=False))
    print("Best cluster model:", best_model_id)
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
