from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path


# ============================================================
# APP
# ============================================================

app = Flask(__name__)


# ============================================================
# DOSYA YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "results"
    / "dashboard_articles.csv"
)

K_RESULT_FILE = (
    BASE_DIR
    / "kmeans_fine_k_results.csv"
)


# ============================================================
# VERİYİ YÜKLE
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig"
)

# Boolean kolonları garantiye al
boolean_columns = [
    "has_doi",
    "has_abstract",
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
                .map({
                    "true": True,
                    "false": False
                })
                .fillna(False)
            )


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value)


def percentage(value):

    return round(
        value * 100,
        2
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    total_articles = len(df)

    stats = {

        "total_articles":
            total_articles,

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
                df["all_levels_found"].sum()
            ),

        "main_match":
            percentage(
                df["main_match"].mean()
            ),

        "sub_match":
            percentage(
                df["sub_match"].mean()
            ),

        "leaf_match":
            percentage(
                df["leaf_match"].mean()
            )
    }


    # --------------------------------------------------------
    # ANA ALAN DAĞILIMI
    # --------------------------------------------------------

    main_distribution = (
        df[
            df["predicted_main_field"]
            .fillna("")
            !=
            ""
        ]
        ["predicted_main_field"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # EN ÇOK TAHMİN EDİLEN ALT ALANLAR
    # --------------------------------------------------------

    sub_distribution = (
        df[
            df["predicted_sub_field"]
            .fillna("")
            !=
            ""
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
# MAKALE LİSTESİ
# ============================================================

@app.route("/articles")
def articles():

    filtered = df.copy()


    # --------------------------------------------------------
    # ARAMA
    # --------------------------------------------------------

    q = request.args.get(
        "q",
        ""
    ).strip()


    if q:

        mask = (

            filtered[
                "title"
            ]
            .fillna("")
            .str.contains(
                q,
                case=False,
                na=False
            )

            |

            filtered[
                "abstract"
            ]
            .fillna("")
            .str.contains(
                q,
                case=False,
                na=False
            )

            |

            filtered[
                "doi"
            ]
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
    # ANA ALAN FİLTRESİ
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
    # ALT ALAN FİLTRESİ
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
    # LEAF DURUMU
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
    # DOI FİLTRESİ
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
    # UYUM FİLTRESİ
    # --------------------------------------------------------

    match_status = request.args.get(
        "match_status",
        ""
    )


    if match_status == "MAIN_MATCH":

        filtered = filtered[
            filtered[
                "main_match"
            ]
        ]

    elif match_status == "MAIN_MISMATCH":

        filtered = filtered[
            ~filtered[
                "main_match"
            ]
        ]

    elif match_status == "SUB_MATCH":

        filtered = filtered[
            filtered[
                "sub_match"
            ]
        ]

    elif match_status == "SUB_MISMATCH":

        filtered = filtered[
            ~filtered[
                "sub_match"
            ]
        ]

    elif match_status == "LEAF_MATCH":

        filtered = filtered[
            filtered[
                "leaf_match"
            ]
        ]

    elif match_status == "LEAF_MISMATCH":

        filtered = filtered[
            ~filtered[
                "leaf_match"
            ]
        ]


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
        page
        -
        1
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
                "article_id":
                    int(
                        row[
                            "article_id"
                        ]
                    ),

                "doi":
                    safe_text(
                        row["doi"]
                    ),

                "title":
                    safe_text(
                        row["title"]
                    ),

                "abstract_preview":
                    abstract_preview,

                "display_language":
                    safe_text(
                        row[
                            "display_language"
                        ]
                    ),

                "predicted_main_field":
                    safe_text(
                        row[
                            "predicted_main_field"
                        ]
                    ),

                "predicted_sub_field":
                    safe_text(
                        row[
                            "predicted_sub_field"
                        ]
                    ),

                "predicted_subject":
                    safe_text(
                        row[
                            "predicted_subject"
                        ]
                    ),

                "main_found":
                    bool(
                        row[
                            "main_found"
                        ]
                    ),

                "sub_found":
                    bool(
                        row[
                            "sub_found"
                        ]
                    ),

                "leaf_found":
                    bool(
                        row[
                            "leaf_found"
                        ]
                    ),

                "main_match":
                    bool(
                        row[
                            "main_match"
                        ]
                    ),

                "sub_match":
                    bool(
                        row[
                            "sub_match"
                        ]
                    ),

                "leaf_match":
                    bool(
                        row[
                            "leaf_match"
                        ]
                    )
            }
        )


    # --------------------------------------------------------
    # DROPDOWN DEĞERLERİ
    # --------------------------------------------------------

    main_fields = sorted(
        [
            value
            for value in
            df[
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
            for value in
            df[
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
# MAKALE DETAYI
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
# K-MEANS ANALİZİ
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