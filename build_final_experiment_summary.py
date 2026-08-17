import pandas as pd
import numpy as np


# ============================================================
# FINAL DENEY ÖZETİ
# ============================================================

rows = []


def add_result(
    experiment,
    precision=None,
    recall=None,
    f1=None,
    hit_rate=None,
    exact_match=None,
    notes=""
):
    rows.append(
        {
            "Experiment": experiment,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Hit_Rate": hit_rate,
            "Exact_Match": exact_match,
            "Notes": notes
        }
    )


# ============================================================
# 1. K-MEANS BASELINE
# ============================================================

add_result(
    "Qwen3 + K-Means Dynamic",
    precision=0.4078,
    recall=0.3490,
    f1=0.3762,
    hit_rate=0.6141,
    notes="1024D Qwen3 embedding, K=195, dynamic multi-cluster"
)


# ============================================================
# 2. SEMANTIC HARD FILTER
# ============================================================

semantic_file = "semantic_subject_validation_results.csv"

try:
    semantic = pd.read_csv(
        semantic_file,
        encoding="utf-8-sig"
    )

    best_semantic = semantic.loc[
        semantic["F1"].idxmax()
    ]

    add_result(
        "K-Means + Semantic Hard Filter",
        precision=best_semantic["Precision"],
        recall=best_semantic["Recall"],
        f1=best_semantic["F1"],
        hit_rate=best_semantic["Hit_Rate"],
        exact_match=best_semantic["Exact_Match"],
        notes=(
            f"Best semantic threshold="
            f"{best_semantic['Semantic_Threshold']}"
        )
    )

except FileNotFoundError:
    print(
        "Bulunamadı:",
        semantic_file
    )


# ============================================================
# 3. FINAL HİBRİT
# ============================================================

hybrid_file = "hybrid_topic_score_results.csv"

try:
    hybrid = pd.read_csv(
        hybrid_file,
        encoding="utf-8-sig"
    )

    best_hybrid = hybrid.loc[
        hybrid["F1"].idxmax()
    ]

    add_result(
        "Final Hybrid K-Means + Semantic",
        precision=best_hybrid["Precision"],
        recall=best_hybrid["Recall"],
        f1=best_hybrid["F1"],
        hit_rate=best_hybrid["Hit_Rate"],
        exact_match=best_hybrid["Exact_Match"],
        notes=(
            f"Alpha={best_hybrid['Alpha']}, "
            f"threshold={best_hybrid['Final_Threshold']}"
        )
    )

except FileNotFoundError:
    print(
        "Bulunamadı:",
        hybrid_file
    )


# ============================================================
# 4. HİYERARŞİK ROOT
# ============================================================

hierarchical_file = "hierarchical_hybrid_results.csv"

try:
    hierarchy = pd.read_csv(
        hierarchical_file,
        encoding="utf-8-sig"
    )

    root_rows = hierarchy[
        hierarchy["Method"]
        ==
        "ROOT"
    ]

    if not root_rows.empty:

        best_root = root_rows.loc[
            root_rows["F1"].idxmax()
        ]

        add_result(
            "Hierarchical Hybrid - ROOT",
            precision=best_root["Precision"],
            recall=best_root["Recall"],
            f1=best_root["F1"],
            hit_rate=best_root["Hit_Rate"],
            exact_match=best_root["Exact_Match"],
            notes=(
                f"Hierarchy margin="
                f"{best_root['Hierarchy_Margin']}"
            )
        )


    # ========================================================
    # 5. HİYERARŞİK LEVEL2
    # ========================================================

    level2_rows = hierarchy[
        hierarchy["Method"]
        ==
        "LEVEL2"
    ]

    if not level2_rows.empty:

        best_level2 = level2_rows.loc[
            level2_rows["F1"].idxmax()
        ]

        add_result(
            "Hierarchical Hybrid - LEVEL2",
            precision=best_level2["Precision"],
            recall=best_level2["Recall"],
            f1=best_level2["F1"],
            hit_rate=best_level2["Hit_Rate"],
            exact_match=best_level2["Exact_Match"],
            notes=(
                f"Hierarchy margin="
                f"{best_level2['Hierarchy_Margin']}"
            )
        )

except FileNotFoundError:
    print(
        "Bulunamadı:",
        hierarchical_file
    )


# ============================================================
# 6. UMAP25
# ============================================================

umap_file = "umap25_euclidean_parameter_results.csv"

try:
    umap = pd.read_csv(
        umap_file,
        encoding="utf-8-sig"
    )

    best_umap = umap.loc[
        umap["F1"].idxmax()
    ]

    add_result(
        "Qwen3 + UMAP25 + K-Means",
        precision=best_umap["Precision"],
        recall=best_umap["Recall"],
        f1=best_umap["F1"],
        hit_rate=best_umap["Hit_Rate"],
        exact_match=best_umap["Exact_Match"],
        notes=(
            f"Distance margin="
            f"{best_umap['Distance_Margin']}, "
            f"subject threshold="
            f"{best_umap['Subject_Threshold']}"
        )
    )

except FileNotFoundError:
    print(
        "Bulunamadı:",
        umap_file
    )


# ============================================================
# 7. SEMANTIC-AWARE 0.70
# ============================================================

semantic_aware_file = (
    "semantic_aware_evaluation_results.csv"
)

try:
    semantic_aware = pd.read_csv(
        semantic_aware_file,
        encoding="utf-8-sig"
    )

    semantic_070 = semantic_aware[
        np.isclose(
            semantic_aware[
                "Semantic_Threshold"
            ],
            0.70
        )
    ]

    if not semantic_070.empty:

        semantic_070 = semantic_070.iloc[0]

        add_result(
            "Final Hybrid - Semantic-Aware Evaluation",
            precision=semantic_070["Precision"],
            recall=semantic_070["Recall"],
            f1=semantic_070["F1"],
            hit_rate=semantic_070["Hit_Rate"],
            exact_match=semantic_070[
                "Semantic_Exact_Match"
            ],
            notes=(
                "Semantic threshold=0.70; "
                "evaluation metric, not exact-label result"
            )
        )

except FileNotFoundError:
    print(
        "Bulunamadı:",
        semantic_aware_file
    )


# ============================================================
# DATAFRAME
# ============================================================

summary = pd.DataFrame(
    rows
)


# ============================================================
# F1 SIRALAMASI
# ============================================================

summary = summary.sort_values(
    "F1",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# EKRAN
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)

pd.set_option(
    "display.max_colwidth",
    100
)


print("\n" + "=" * 150)
print("FINAL PROJE DENEY KARŞILAŞTIRMASI")
print("=" * 150)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL KARARLAR
# ============================================================

print("\n" + "=" * 120)
print("FINAL SİSTEM KARARI")
print("=" * 120)

print(
    "Embedding modeli:"
    " Qwen/Qwen3-Embedding-0.6B"
)

print(
    "Embedding boyutu:"
    " 1024"
)

print(
    "Clustering:"
    " K-Means"
)

print(
    "K:"
    " 195"
)

print(
    "Final hibrit:"
    " 0.55 * KMeansScore"
    " + 0.45 * SemanticSimilarity"
)

print(
    "Final threshold:"
    " 0.26"
)

print(
    "Exact-label F1:"
    " 0.3828"
)

print(
    "Semantic-aware threshold:"
    " 0.70"
)

print(
    "Semantic-aware F1:"
    " 0.5569"
)


# ============================================================
# CSV
# ============================================================

summary.to_csv(
    "final_experiment_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDosya oluşturuldu:"
    " final_experiment_summary.csv"
)