from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path


app = Flask(__name__)


BASE_DIR = Path(
    __file__
).resolve().parent.parent


DATA_FILE = (
    BASE_DIR
    / "results"
    / "dashboard_articles.csv"
)

K_RESULT_FILE = (
    BASE_DIR
    / "kmeans_fine_k_results.csv"
)


df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# BOOL KOLONLAR
# ============================================================

boolean_columns = [
    "has_doi",
    "has_abstract",
    "has_trdizin_subject",

    "main_found",
    "sub_found",
    "leaf_found",

    "main_match",
    "sub_match",
    "leaf_match",

    "all_levels_found",
    "all_levels_match"
]


for column in boolean_columns:

    if column in df.columns:

        if df[column].dtype != bool:

            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False
                    }
                )
                .fillna(False)
            )


def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value)


def percentage(value):

    if pd.isna(value):
        return 0

    return round(
        value * 100,
        2
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    labeled = df[
        df["has_trdizin_subject"]
    ].copy()


    stats = {

        "total_articles":
            len(df),

        "doi_found":
            int(
                df["has_doi"].sum()
            ),

        "doi_missing":
            int(
                (~df["has_doi"]).sum()
            ),

        "abstract_found":
            int(
                df["has_abstract"].sum()
            ),

        "abstract_missing":
            int(
                (~df["has_abstract"]).sum()
            ),

        "trdizin_found":
            int(
                df[
                    "has_trdizin_subject"
                ].sum()
            ),

        "trdizin_missing":
            int(
                (
                    ~df[
                        "has_trdizin_subject"
                    ]
                ).sum()
            ),

        "main_found":
            int(
                df["main_found"].sum()
            ),

        "main_missing":
            int(
                (~df["main_found"]).sum()
            ),

        "sub_found":
            int(
                df["sub_found"].sum()
            ),

        "sub_missing":
            int(
                (~df["sub_found"]).sum()
            ),

        "leaf_found":
            int(
                df["leaf_found"].sum()
            ),

        "leaf_missing":
            int(
                (~df["leaf_found"]).sum()
            ),

        "all_levels_found":
            int(
                df[
                    "all_levels_found"
                ].sum()
            ),

        # SADECE ETİKETLİLERDE
        "evaluated_articles":
            len(labeled),

        "not_evaluated":
            len(df)
            -
            len(labeled),

        "main_match":
            percentage(
                labeled[
                    "main_match"
                ].mean()
            ),

        "sub_match":
            percentage(
                labeled[
                    "sub_match"
                ].mean()
            ),

        "leaf_match":
            percentage(
                labeled[
                    "leaf_match"
                ].mean()
            ),

        "main_match_count":
            int(
                (
                    labeled[
                        "main_comparison_status"
                    ]
                    ==
                    "UYUMLU"
                ).sum()
            ),

        "main_mismatch_count":
            int(
                (
                    labeled[
                        "main_comparison_status"
                    ]
                    ==
                    "UYUSMUYOR"
                ).sum()
            ),

        "sub_match_count":
            int(
                (
                    labeled[
                        "sub_comparison_status"
                    ]
                    ==
                    "UYUMLU"
                ).sum()
            ),

        "sub_mismatch_count":
            int(
                (
                    labeled[
                        "sub_comparison_status"
                    ]
                    ==
                    "UYUSMUYOR"
                ).sum()
            ),

        "leaf_match_count":
            int(
                (
                    labeled[
                        "leaf_comparison_status"
                    ]
                    ==
                    "UYUMLU"
                ).sum()
            ),

        "leaf_mismatch_count":
            int(
                (
                    labeled[
                        "leaf_comparison_status"
                    ]
                    ==
                    "UYUSMUYOR"
                ).sum()
            ),

        "semantic_support":
            round(
                df[
                    "top5_mean_similarity"
                ].mean(),
                4
            )
    }


    main_distribution = (
        df[
            df[
                "predicted_main_field"
            ]
            .fillna("")
            != ""
        ]
        ["predicted_main_field"]
        .value_counts()
        .to_dict()
    )


    sub_distribution = (
        df[
            df[
                "predicted_sub_field"
            ]
            .fillna("")
            != ""
        ]
        ["predicted_sub_field"]
        .value_counts()
        .head(10)
        .to_dict()
    )


    return render_template(
        "dashboard.html",

        stats=stats,

        main_distribution=
            main_distribution,

        sub_distribution=
            sub_distribution
    )


# ============================================================
# MAKALELER
# ============================================================

@app.route("/articles")
def articles():

    filtered = df.copy()


    q = request.args.get(
        "q",
        ""
    ).strip()


    if q:

        mask = (

            filtered["title"]
            .fillna("")
            .str.contains(
                q,
                case=False,
                na=False
            )

            |

            filtered["abstract"]
            .fillna("")
            .str.contains(
                q,
                case=False,
                na=False
            )

            |

            filtered["doi"]
            .fillna("")
            .str.contains(
                q,
                case=False,
                na=False
            )
        )

        filtered = filtered[
            mask
        ]


    # --------------------------------------------------------
    # TR DİZİN ETİKET
    # --------------------------------------------------------

    trdizin_status = request.args.get(
        "trdizin_status",
        ""
    )


    if trdizin_status == "FOUND":

        filtered = filtered[
            filtered[
                "has_trdizin_subject"
            ]
        ]

    elif trdizin_status == "MISSING":

        filtered = filtered[
            ~filtered[
                "has_trdizin_subject"
            ]
        ]


    # --------------------------------------------------------
    # ANA ALAN
    # --------------------------------------------------------

    main_field = request.args.get(
        "main_field",
        ""
    )


    if main_field:

        if main_field == "MISSING":

            filtered = filtered[
                ~filtered[
                    "main_found"
                ]
            ]

        else:

            filtered = filtered[
                filtered[
                    "predicted_main_field"
                ]
                ==
                main_field
            ]


    # --------------------------------------------------------
    # ALT ALAN
    # --------------------------------------------------------

    sub_field = request.args.get(
        "sub_field",
        ""
    )


    if sub_field:

        if sub_field == "MISSING":

            filtered = filtered[
                ~filtered[
                    "sub_found"
                ]
            ]

        else:

            filtered = filtered[
                filtered[
                    "predicted_sub_field"
                ]
                ==
                sub_field
            ]


    # --------------------------------------------------------
    # LEAF
    # --------------------------------------------------------

    leaf_status = request.args.get(
        "leaf_status",
        ""
    )


    if leaf_status == "FOUND":

        filtered = filtered[
            filtered[
                "leaf_found"
            ]
        ]

    elif leaf_status == "MISSING":

        filtered = filtered[
            ~filtered[
                "leaf_found"
            ]
        ]


    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi_status = request.args.get(
        "doi_status",
        ""
    )


    if doi_status == "FOUND":

        filtered = filtered[
            filtered[
                "has_doi"
            ]
        ]

    elif doi_status == "MISSING":

        filtered = filtered[
            ~filtered[
                "has_doi"
            ]
        ]


    # --------------------------------------------------------
    # UYUM
    # --------------------------------------------------------

    match_status = request.args.get(
        "match_status",
        ""
    )


    if match_status == "MAIN_MATCH":

        filtered = filtered[
            filtered[
                "main_comparison_status"
            ]
            ==
            "UYUMLU"
        ]

    elif match_status == "MAIN_MISMATCH":

        filtered = filtered[
            filtered[
                "main_comparison_status"
            ]
            ==
            "UYUSMUYOR"
        ]

    elif match_status == "SUB_MATCH":

        filtered = filtered[
            filtered[
                "sub_comparison_status"
            ]
            ==
            "UYUMLU"
        ]

    elif match_status == "SUB_MISMATCH":

        filtered = filtered[
            filtered[
                "sub_comparison_status"
            ]
            ==
            "UYUSMUYOR"
        ]

    elif match_status == "LEAF_MATCH":

        filtered = filtered[
            filtered[
                "leaf_comparison_status"
            ]
            ==
            "UYUMLU"
        ]

    elif match_status == "LEAF_MISMATCH":

        filtered = filtered[
            filtered[
                "leaf_comparison_status"
            ]
            ==
            "UYUSMUYOR"
        ]

    elif match_status == "NOT_EVALUATED":

        filtered = filtered[
            ~filtered[
                "has_trdizin_subject"
            ]
        ]


    # Etiketliler önce
    filtered = filtered.sort_values(
        [
            "has_trdizin_subject",
            "article_id"
        ],
        ascending=[
            False,
            True
        ]
    )


    # --------------------------------------------------------
    # SAYFALAMA
    # --------------------------------------------------------

    page = request.args.get(
        "page",
        1,
        type=int
    )

    per_page = 25

    total_filtered = len(
        filtered
    )

    total_pages = max(
        1,
        (
            total_filtered
            +
            per_page
            -
            1
        )
        //
        per_page
    )

    page = max(
        1,
        min(
            page,
            total_pages
        )
    )


    start = (
        page - 1
    ) * per_page

    end = (
        start
        +
        per_page
    )


    page_df = filtered.iloc[
        start:end
    ].copy()


    rows = []


    for _, row in page_df.iterrows():

        abstract = safe_text(
            row["abstract"]
        )

        if len(abstract) > 220:

            abstract_preview = (
                abstract[:220]
                +
                "..."
            )

        else:

            abstract_preview = abstract


        rows.append(
            {
                key:
                    (
                        ""
                        if pd.isna(value)
                        else value
                    )

                for key, value
                in row.to_dict().items()
            }
        )

        rows[-1][
            "abstract_preview"
        ] = abstract_preview


    main_fields = sorted(
        [
            value
            for value
            in df[
                "predicted_main_field"
            ]
            .dropna()
            .unique()

            if str(value).strip()
        ]
    )


    sub_fields = sorted(
        [
            value
            for value
            in df[
                "predicted_sub_field"
            ]
            .dropna()
            .unique()

            if str(value).strip()
        ]
    )


    return render_template(
        "articles.html",

        articles=rows,

        total_filtered=
            total_filtered,

        page=
            page,

        total_pages=
            total_pages,

        main_fields=
            main_fields,

        sub_fields=
            sub_fields
    )


# ============================================================
# DETAY
# ============================================================

@app.route(
    "/article/<int:article_id>"
)
def article_detail(
    article_id
):

    match = df[
        df["article_id"]
        ==
        article_id
    ]


    if match.empty:

        return (
            "Makale bulunamadı",
            404
        )


    row = match.iloc[0]


    article = {

        key:
            (
                ""
                if pd.isna(value)
                else value
            )

        for key, value
        in row.to_dict().items()
    }


    return render_template(
        "article_detail.html",
        article=article
    )


# ============================================================
# K ANALİZİ
# ============================================================

@app.route("/k-analysis")
def k_analysis():

    if not K_RESULT_FILE.exists():

        return (
            "K sonuç dosyası bulunamadı.",
            404
        )


    k_df = pd.read_csv(
        K_RESULT_FILE,
        encoding="utf-8-sig"
    )


    k_data = (
        k_df
        .sort_values("K")
        .to_dict(
            orient="records"
        )
    )


    return render_template(
        "k_analysis.html",
        k_data=k_data
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )