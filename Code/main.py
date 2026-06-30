import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pymongo
from pymongo import MongoClient
from datetime import datetime
import os


CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "air_quality_db"
COLLECTION_NAME = "readings"

SAFE_LIMITS = {
    "PM2.5": 60,
    "PM10":  100,
    "NO2":   80,
    "SO2":   80,
    "CO":    4000,
    "OZONE": 100,
    "NH3":   400,
}

COLORS = {
    "bg":         "#080f1e",
    "bg2":        "#0d1829",
    "card":       "#111e33",
    "card2":      "#162440",
    "accent":     "#38bdf8",
    "accent2":    "#818cf8",
    "accent3":    "#34d399",
    "good":       "#34d399",
    "moderate":   "#fbbf24",
    "bad":        "#f87171",
    "text":       "#f0f6ff",
    "subtext":    "#7a93b8",
    "border":     "#1e3354",
    "border2":    "#243d5c",
    "highlight":  "#1a3050",
}

FONT_MAIN = "Segoe UI"
FONT_MONO = "Consolas"

# ─────────────────────────────────────────────
# DATABASE LAYER  (MongoDB CRUD)
# ─────────────────────────────────────────────
class Database:
    def __init__(self):
        self.connected = False
        self.client = None
        self.db = None
        self.col = None
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            self.client.server_info()
            self.db  = self.client[DB_NAME]
            self.col = self.db[COLLECTION_NAME]
            self.connected = True
        except Exception:
            self.connected = False

    def insert_many(self, records: list):
        if not self.connected:
            return 0
        self.col.delete_many({})
        result = self.col.insert_many(records)
        return len(result.inserted_ids)

    def find(self, query: dict = None, limit: int = 0) -> list:
        if not self.connected:
            return []
        cursor = self.col.find(query or {}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def update_record(self, station: str, pollutant: str, new_avg: float):
        if not self.connected:
            return False
        result = self.col.update_one(
            {"station": station, "pollutant_id": pollutant},
            {"$set": {"pollutant_avg": new_avg, "updated_at": datetime.now().isoformat()}}
        )
        return result.modified_count > 0

    def delete_city(self, city: str) -> int:
        if not self.connected:
            return 0
        result = self.col.delete_many({"city": city})
        return result.deleted_count

    def count(self) -> int:
        if not self.connected:
            return 0
        return self.col.count_documents({})


# ─────────────────────────────────────────────
# DATA LAYER  (Pandas + NumPy)
# ─────────────────────────────────────────────
class DataProcessor:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._clean()

    def _clean(self):
        self.df.dropna(subset=["pollutant_avg", "state", "city"], inplace=True)
        for col in ["pollutant_avg", "pollutant_min", "pollutant_max"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df.dropna(subset=["pollutant_avg"], inplace=True)
        self.df["state"] = self.df["state"].str.replace("_", " ")

    def numpy_stats(self, series: pd.Series) -> dict:
        arr = np.array(series.dropna())
        return {
            "mean":          round(float(np.mean(arr)), 2),
            "median":        round(float(np.median(arr)), 2),
            "std":           round(float(np.std(arr)), 2),
            "min":           round(float(np.min(arr)), 2),
            "max":           round(float(np.max(arr)), 2),
            "percentile_75": round(float(np.percentile(arr, 75)), 2),
            "percentile_95": round(float(np.percentile(arr, 95)), 2),
        }

    def get_states(self):
        return sorted(self.df["state"].unique().tolist())

    def get_cities(self, state=None):
        if state:
            return sorted(self.df[self.df["state"] == state]["city"].unique().tolist())
        return sorted(self.df["city"].unique().tolist())

    def get_pollutants(self):
        return sorted(self.df["pollutant_id"].unique().tolist())

    def filter(self, state=None, city=None, pollutant=None):
        df = self.df.copy()
        if state:     df = df[df["state"]        == state]
        if city:      df = df[df["city"]          == city]
        if pollutant: df = df[df["pollutant_id"]  == pollutant]
        return df

    def state_avg(self, pollutant):
        return (self.df[self.df["pollutant_id"] == pollutant]
                .groupby("state")["pollutant_avg"].mean()
                .reset_index().sort_values("pollutant_avg", ascending=False))

    def top_polluted_cities(self, pollutant, n=10):
        return (self.df[self.df["pollutant_id"] == pollutant]
                .groupby("city")["pollutant_avg"].mean()
                .reset_index().sort_values("pollutant_avg", ascending=False).head(n))

    def to_records(self):
        return self.df.to_dict("records")

    def aqi_health_label(self, value, pollutant):
        limit = SAFE_LIMITS.get(pollutant, 100)
        ratio = value / limit
        if ratio <= 0.5:   return "Good",      COLORS["good"]
        elif ratio <= 1.0: return "Moderate",  COLORS["moderate"]
        else:              return "Unhealthy",  COLORS["bad"]


# ─────────────────────────────────────────────
# CHART STYLE
# ─────────────────────────────────────────────
def apply_chart_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["card"],
        "axes.facecolor":    COLORS["bg2"],
        "axes.edgecolor":    COLORS["border2"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["subtext"],
        "ytick.color":       COLORS["subtext"],
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def make_card(parent, **kw):
    return tk.Frame(parent, bg=COLORS["card"],
                    highlightbackground=COLORS["border2"],
                    highlightthickness=1, **kw)


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
class AirQualityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Air Quality Monitoring and Analysis System")
        self.geometry("1320x800")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.minsize(1100, 680)

        apply_chart_style()

        self.dp = DataProcessor(CSV_PATH)
        self.db = Database()

        if self.db.connected:
            count = self.db.insert_many(self.dp.to_records())
            self.db_status = f"MongoDB ✓  {count} docs"
            self.db_ok = True
        else:
            self.db_status = "MongoDB ✗  offline"
            self.db_ok = False

        self._build_ui()

    # ── HEADER ───────────────────────────────
    def _build_ui(self):
        # Top accent line
        tk.Frame(self, bg=COLORS["accent"], height=3).pack(fill="x")

        hdr = tk.Frame(self, bg=COLORS["card2"], pady=10)
        hdr.pack(fill="x")

        left = tk.Frame(hdr, bg=COLORS["card2"])
        left.pack(side="left", padx=20)
        tk.Label(left, text="🌿", font=(FONT_MAIN, 22),
                 bg=COLORS["card2"], fg=COLORS["accent"]).pack(side="left", padx=(0, 12))
        title_block = tk.Frame(left, bg=COLORS["card2"])
        title_block.pack(side="left")
        tk.Label(title_block,
                 text="Air Quality Monitoring and Analysis System",
                 font=(FONT_MAIN, 15, "bold"),
                 bg=COLORS["card2"], fg=COLORS["text"]).pack(anchor="w")
        tk.Label(title_block,
                 text="India  •  30 States  •  241 Cities  •  7 Pollutants",
                 font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(anchor="w")

        right = tk.Frame(hdr, bg=COLORS["card2"])
        right.pack(side="right", padx=20)
        pill_clr = COLORS["good"] if self.db_ok else COLORS["bad"]
        pill_bg  = "#0d2e1a" if self.db_ok else "#2e0d0d"
        pill = tk.Frame(right, bg=pill_bg,
                        highlightbackground=pill_clr, highlightthickness=1)
        pill.pack()
        tk.Label(pill, text=f"  {self.db_status}  ",
                 font=(FONT_MAIN, 9, "bold"),
                 bg=pill_bg, fg=pill_clr).pack(pady=5, padx=4)

        # Notebook
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=COLORS["bg"], borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["card2"],
                        foreground=COLORS["subtext"],
                        padding=[18, 8],
                        font=(FONT_MAIN, 10),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["bg"]),
                               ("active",   COLORS["highlight"])],
                  foreground=[("selected", COLORS["accent"]),
                               ("active",   COLORS["text"])])

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self._tab_overview()
        self._tab_explore()
        self._tab_charts()
        self._tab_stats()
        self._tab_crud()

    # ── TAB 1 : OVERVIEW ─────────────────────
    def _tab_overview(self):
        tab = tk.Frame(self.nb, bg=COLORS["bg"])
        self.nb.add(tab, text="  📊  Overview  ")

        kpi_row = tk.Frame(tab, bg=COLORS["bg"])
        kpi_row.pack(fill="x", padx=20, pady=(18, 8))

        kpis = [
            ("Total Records",  str(len(self.dp.df)),                     COLORS["accent"],  "📋"),
            ("States Covered", str(self.dp.df["state"].nunique()),        COLORS["accent2"], "🗺️"),
            ("Cities Covered", str(self.dp.df["city"].nunique()),         COLORS["good"],    "🏙️"),
            ("Pollutants",     str(self.dp.df["pollutant_id"].nunique()), COLORS["moderate"],"☁️"),
        ]
        for title, val, clr, icon in kpis:
            card = tk.Frame(kpi_row, bg=COLORS["card"],
                            highlightbackground=clr, highlightthickness=1)
            card.pack(side="left", expand=True, fill="x", padx=8)
            tk.Frame(card, bg=clr, height=3).pack(fill="x")
            top = tk.Frame(card, bg=COLORS["card"])
            top.pack(fill="x", padx=14, pady=(10, 0))
            tk.Label(top, text=icon, font=(FONT_MAIN, 18),
                     bg=COLORS["card"], fg=clr).pack(side="left")
            tk.Label(top, text=val,  font=(FONT_MAIN, 28, "bold"),
                     bg=COLORS["card"], fg=clr).pack(side="right")
            tk.Label(card, text=title, font=(FONT_MAIN, 10),
                     bg=COLORS["card"], fg=COLORS["subtext"]).pack(pady=(4, 12))

        chart_card = make_card(tab)
        chart_card.pack(fill="both", expand=True, padx=20, pady=(4, 16))

        hdr = tk.Frame(chart_card, bg=COLORS["card2"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Top 10 Most Polluted Cities — PM2.5",
                 font=(FONT_MAIN, 11, "bold"),
                 bg=COLORS["card2"], fg=COLORS["text"]).pack(side="left", pady=9)
        tk.Label(hdr, text=f"  Safe Limit: {SAFE_LIMITS['PM2.5']} µg/m³  ",
                 font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="right", pady=9)

        top = self.dp.top_polluted_cities("PM2.5", 10)
        fig = Figure(figsize=(12, 4.2), dpi=96)
        ax  = fig.add_subplot(111)

        bar_colors = [COLORS["bad"] if v > SAFE_LIMITS["PM2.5"] else COLORS["moderate"]
                      for v in top["pollutant_avg"]]
        bars = ax.barh(top["city"], top["pollutant_avg"],
                       color=bar_colors, edgecolor="none", height=0.6)
        for bar, val in zip(bars, top["pollutant_avg"]):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}", va="center", ha="left",
                    fontsize=9, color=COLORS["subtext"])
        ax.axvline(SAFE_LIMITS["PM2.5"], color=COLORS["good"],
                   linestyle="--", linewidth=1.5,
                   label=f"Safe Limit ({SAFE_LIMITS['PM2.5']})")
        ax.set_xlabel("PM2.5 Average (µg/m³)", labelpad=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["border2"],
                  labelcolor=COLORS["text"], fontsize=9)
        ax.tick_params(labelsize=9)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=chart_card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # ── TAB 2 : EXPLORE ──────────────────────
    def _tab_explore(self):
        tab = tk.Frame(self.nb, bg=COLORS["bg"])
        self.nb.add(tab, text="  🔍  Explore  ")

        ctrl = tk.Frame(tab, bg=COLORS["card2"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        ctrl.pack(fill="x", padx=16, pady=(14, 4))
        inner = tk.Frame(ctrl, bg=COLORS["card2"])
        inner.pack(fill="x", padx=14, pady=10)

        def lbl(text):
            tk.Label(inner, text=text, font=(FONT_MAIN, 9),
                     bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="left", padx=(10, 3))

        lbl("State")
        self._exp_state = tk.StringVar(value="All")
        state_cb = ttk.Combobox(inner, textvariable=self._exp_state,
                                width=20, values=["All"] + self.dp.get_states(),
                                state="readonly")
        state_cb.pack(side="left", padx=4)

        lbl("City")
        self._exp_city = tk.StringVar(value="All")
        self._city_cb  = ttk.Combobox(inner, textvariable=self._exp_city,
                                      width=20, values=["All"], state="readonly")
        self._city_cb.pack(side="left", padx=4)

        lbl("Pollutant")
        self._exp_poll = tk.StringVar(value="All")
        ttk.Combobox(inner, textvariable=self._exp_poll, width=10,
                     values=["All"] + self.dp.get_pollutants(),
                     state="readonly").pack(side="left", padx=4)

        tk.Button(inner, text="Apply Filter", command=self._apply_explore_filter,
                  bg=COLORS["accent"], fg=COLORS["bg"],
                  font=(FONT_MAIN, 9, "bold"), relief="flat",
                  cursor="hand2", padx=14, pady=5).pack(side="left", padx=12)
        tk.Button(inner, text="Reset", command=self._reset_explore,
                  bg=COLORS["border2"], fg=COLORS["text"],
                  font=(FONT_MAIN, 9), relief="flat",
                  cursor="hand2", padx=10, pady=5).pack(side="left")

        self._exp_count_lbl = tk.Label(inner, text="", font=(FONT_MAIN, 9, "bold"),
                                       bg=COLORS["card2"], fg=COLORS["accent"])
        self._exp_count_lbl.pack(side="right", padx=10)

        def on_state_change(*_):
            st = self._exp_state.get()
            cities = ["All"] + (self.dp.get_cities(st) if st != "All" else self.dp.get_cities())
            self._city_cb["values"] = cities
            self._exp_city.set("All")
        state_cb.bind("<<ComboboxSelected>>", on_state_change)

        # Legend
        leg = tk.Frame(tab, bg=COLORS["bg"])
        leg.pack(fill="x", padx=16, pady=(2, 4))
        for txt, fg, bg in [("  ● Good  ", COLORS["good"],     "#0e2a1a"),
                              ("  ● Moderate  ", COLORS["moderate"], "#2e2a0a"),
                              ("  ● Unhealthy  ", COLORS["bad"],  "#2e1010")]:
            tk.Label(leg, text=txt, font=(FONT_MAIN, 9),
                     bg=bg, fg=fg).pack(side="left", padx=4, pady=2)

        tbl_card = make_card(tab)
        tbl_card.pack(fill="both", expand=True, padx=16, pady=(2, 14))

        cols = ["state", "city", "station", "pollutant_id",
                "pollutant_min", "pollutant_avg", "pollutant_max"]
        self._exp_tree = ttk.Treeview(tbl_card, columns=cols, show="headings", height=22)

        for c, w, anc in zip(cols,
                              [130, 110, 280, 100, 90, 90, 90],
                              ["w","w","w","center","center","center","center"]):
            self._exp_tree.heading(c, text=c.replace("_"," ").title())
            self._exp_tree.column(c, width=w, anchor=anc, minwidth=60)

        vsb = ttk.Scrollbar(tbl_card, orient="vertical",   command=self._exp_tree.yview)
        hsb = ttk.Scrollbar(tbl_card, orient="horizontal", command=self._exp_tree.xview)
        self._exp_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._exp_tree.pack(fill="both", expand=True)

        self._style_tree()
        self._apply_explore_filter()

    def _style_tree(self):
        s = ttk.Style()
        s.configure("Treeview",
                    background=COLORS["card"],
                    foreground=COLORS["text"],
                    fieldbackground=COLORS["card"],
                    rowheight=28,
                    font=(FONT_MAIN, 9))
        s.configure("Treeview.Heading",
                    background=COLORS["card2"],
                    foreground=COLORS["accent"],
                    font=(FONT_MAIN, 9, "bold"),
                    relief="flat")
        s.map("Treeview",
              background=[("selected", COLORS["highlight"])],
              foreground=[("selected", COLORS["text"])])
        self._exp_tree.tag_configure("unhealthy", background="#2e1010", foreground="#ffaaaa")
        self._exp_tree.tag_configure("moderate",  background="#2e2a0a", foreground="#ffe999")
        self._exp_tree.tag_configure("good",      background="#0e2a1a", foreground="#a0f0c0")

    def _populate_tree(self, df):
        for item in self._exp_tree.get_children():
            self._exp_tree.delete(item)
        cols = ["state","city","station","pollutant_id",
                "pollutant_min","pollutant_avg","pollutant_max"]
        for _, row in df.iterrows():
            vals  = [row.get(c, "") for c in cols]
            label, _ = self.dp.aqi_health_label(row.get("pollutant_avg", 0),
                                                 row.get("pollutant_id", "PM2.5"))
            self._exp_tree.insert("", "end", values=vals, tags=(label.lower(),))

    def _apply_explore_filter(self):
        st   = self._exp_state.get(); st   = None if st   == "All" else st
        city = self._exp_city.get();  city = None if city == "All" else city
        poll = self._exp_poll.get();  poll = None if poll == "All" else poll
        df   = self.dp.filter(st, city, poll)
        self._populate_tree(df)
        self._exp_count_lbl.config(text=f"{len(df):,} records")

    def _reset_explore(self):
        self._exp_state.set("All")
        self._exp_city.set("All")
        self._exp_poll.set("All")
        self._apply_explore_filter()

    # ── TAB 3 : CHARTS ───────────────────────
    def _tab_charts(self):
        tab = tk.Frame(self.nb, bg=COLORS["bg"])
        self.nb.add(tab, text="  📈  Charts  ")

        ctrl = tk.Frame(tab, bg=COLORS["card2"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        ctrl.pack(fill="x", padx=16, pady=(14, 6))
        inner = tk.Frame(ctrl, bg=COLORS["card2"])
        inner.pack(fill="x", padx=14, pady=10)

        tk.Label(inner, text="Chart Type", font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="left", padx=(4, 4))
        self._chart_type = tk.StringVar(value="State Avg Bar")
        ttk.Combobox(inner, textvariable=self._chart_type, width=28,
                     values=["State Avg Bar", "Pollutant Distribution (Box)",
                             "Top Cities (Bar)", "State Heatmap",
                             "Pollutant Pie (Delhi)"],
                     state="readonly").pack(side="left", padx=4)

        tk.Label(inner, text="Pollutant", font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="left", padx=(14, 4))
        self._chart_poll = tk.StringVar(value="PM2.5")
        ttk.Combobox(inner, textvariable=self._chart_poll,
                     values=self.dp.get_pollutants(),
                     width=10, state="readonly").pack(side="left", padx=4)

        tk.Button(inner, text="  Generate Chart  ", command=self._generate_chart,
                  bg=COLORS["accent"], fg=COLORS["bg"],
                  font=(FONT_MAIN, 9, "bold"), relief="flat",
                  cursor="hand2", pady=5).pack(side="left", padx=16)

        self._chart_frame = make_card(tab)
        self._chart_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self._generate_chart()

    def _generate_chart(self):
        for w in self._chart_frame.winfo_children():
            w.destroy()

        ctype = self._chart_type.get()
        poll  = self._chart_poll.get()

        fig = Figure(figsize=(12, 5.2), dpi=96)
        ax  = fig.add_subplot(111)

        if ctype == "State Avg Bar":
            data   = self.dp.state_avg(poll)
            colors = [COLORS["bad"] if v > SAFE_LIMITS.get(poll, 100)
                      else COLORS["good"] for v in data["pollutant_avg"]]
            ax.bar(data["state"], data["pollutant_avg"],
                   color=colors, edgecolor="none", width=0.65)
            ax.axhline(SAFE_LIMITS.get(poll, 100), color=COLORS["moderate"],
                       linestyle="--", linewidth=1.5, label="Safe Limit")
            ax.set_title(f"Average {poll} Concentration by State",
                         fontsize=12, fontweight="bold", pad=12)
            ax.set_ylabel(f"{poll} (µg/m³)")
            ax.tick_params(axis="x", rotation=72, labelsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["border2"],
                      labelcolor=COLORS["text"])

        elif ctype == "Pollutant Distribution (Box)":
            polls    = self.dp.get_pollutants()
            data_box = [self.dp.df[self.dp.df["pollutant_id"] == p]["pollutant_avg"].dropna().values
                        for p in polls]
            clrs = [COLORS["good"], COLORS["accent"], COLORS["accent2"],
                    COLORS["bad"], COLORS["moderate"], COLORS["good"], COLORS["accent2"]]
            bp = ax.boxplot(data_box, labels=polls, patch_artist=True,
                            medianprops=dict(color=COLORS["text"], linewidth=2),
                            whiskerprops=dict(color=COLORS["subtext"]),
                            capprops=dict(color=COLORS["subtext"]),
                            flierprops=dict(marker="o", markersize=3,
                                           markerfacecolor=COLORS["bad"], alpha=0.5))
            for patch, c in zip(bp["boxes"], clrs):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax.set_title("Pollutant Concentration Distribution — All India",
                         fontsize=12, fontweight="bold", pad=12)
            ax.set_ylabel("Average Concentration")
            ax.grid(axis="y", alpha=0.3)

        elif ctype == "Top Cities (Bar)":
            top = self.dp.top_polluted_cities(poll, 15)
            bar_colors = [COLORS["bad"] if v > SAFE_LIMITS.get(poll, 100)
                          else COLORS["moderate"] for v in top["pollutant_avg"]]
            bars = ax.barh(top["city"], top["pollutant_avg"],
                           color=bar_colors, edgecolor="none", height=0.65)
            for bar, val in zip(bars, top["pollutant_avg"]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.0f}", va="center", fontsize=8, color=COLORS["subtext"])
            ax.axvline(SAFE_LIMITS.get(poll, 100), color=COLORS["good"],
                       linestyle="--", linewidth=1.5, label="Safe Limit")
            ax.invert_yaxis()
            ax.set_title(f"Top 15 Most Polluted Cities — {poll}",
                         fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel(f"{poll} (µg/m³)")
            ax.grid(axis="x", alpha=0.3)
            ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["border2"],
                      labelcolor=COLORS["text"])

        elif ctype == "State Heatmap":
            pivot = self.dp.df.pivot_table(index="state", columns="pollutant_id",
                                           values="pollutant_avg", aggfunc="mean").fillna(0)
            norm  = pivot.div(pivot.max(axis=0))
            im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=40, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Normalized Value", color=COLORS["subtext"], fontsize=9)
            ax.set_title("State × Pollutant Heatmap (Normalized)",
                         fontsize=12, fontweight="bold", pad=12)

        elif ctype == "Pollutant Pie (Delhi)":
            delhi = self.dp.df[self.dp.df["city"] == "Delhi"]
            if delhi.empty:
                delhi = self.dp.df[self.dp.df["state"] == "Delhi"]
            if not delhi.empty:
                grp = delhi.groupby("pollutant_id")["pollutant_avg"].mean()
                pie_colors = [COLORS["bad"], COLORS["moderate"], COLORS["accent"],
                              COLORS["accent2"], COLORS["good"], COLORS["subtext"],
                              COLORS["border2"]]
                wedges, texts, autotexts = ax.pie(
                    grp.values, labels=grp.index, autopct="%1.1f%%",
                    colors=pie_colors[:len(grp)],
                    textprops={"color": COLORS["text"], "fontsize": 10},
                    wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
                    startangle=140, pctdistance=0.78)
                for at in autotexts:
                    at.set_fontsize(9)
                    at.set_color(COLORS["bg"])
                ax.set_title("Pollutant Contribution — Delhi",
                             fontsize=12, fontweight="bold", pad=12)
            else:
                ax.text(0.5, 0.5, "Delhi data not found", ha="center", va="center",
                        transform=ax.transAxes, color=COLORS["text"], fontsize=13)

        fig.tight_layout(pad=1.8)
        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # ── TAB 4 : STATISTICS ───────────────────
    def _tab_stats(self):
        tab = tk.Frame(self.nb, bg=COLORS["bg"])
        self.nb.add(tab, text="  🔢  Statistics  ")

        ctrl = tk.Frame(tab, bg=COLORS["card2"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        ctrl.pack(fill="x", padx=16, pady=(14, 6))
        inner = tk.Frame(ctrl, bg=COLORS["card2"])
        inner.pack(fill="x", padx=14, pady=10)

        tk.Label(inner, text="Select Pollutant", font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="left", padx=(4, 6))
        self._stat_poll = tk.StringVar(value="PM2.5")
        ttk.Combobox(inner, textvariable=self._stat_poll,
                     values=self.dp.get_pollutants(),
                     width=12, state="readonly").pack(side="left", padx=4)
        tk.Button(inner, text="  Compute Stats  ", command=self._compute_stats,
                  bg=COLORS["accent2"], fg=COLORS["bg"],
                  font=(FONT_MAIN, 9, "bold"), relief="flat",
                  cursor="hand2", pady=5).pack(side="left", padx=12)

        self._stats_frame = tk.Frame(tab, bg=COLORS["bg"])
        self._stats_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self._compute_stats()

    def _compute_stats(self):
        for w in self._stats_frame.winfo_children():
            w.destroy()

        poll   = self._stat_poll.get()
        series = self.dp.df[self.dp.df["pollutant_id"] == poll]["pollutant_avg"]
        stats  = self.dp.numpy_stats(series)
        limit  = SAFE_LIMITS.get(poll, 100)

        cards_row = tk.Frame(self._stats_frame, bg=COLORS["bg"])
        cards_row.pack(fill="x", pady=(8, 10))

        stat_items = [
            ("Mean",   stats["mean"],          COLORS["accent"],  "μ"),
            ("Median", stats["median"],         COLORS["accent2"], "~"),
            ("Std Dev",stats["std"],            COLORS["moderate"],"σ"),
            ("Min",    stats["min"],            COLORS["good"],    "↓"),
            ("Max",    stats["max"],            COLORS["bad"],     "↑"),
            ("P75",    stats["percentile_75"],  COLORS["accent"],  "75"),
            ("P95",    stats["percentile_95"],  COLORS["bad"],     "95"),
        ]
        for name, val, clr, sym in stat_items:
            card = tk.Frame(cards_row, bg=COLORS["card"],
                            highlightbackground=clr, highlightthickness=1)
            card.pack(side="left", expand=True, fill="x", padx=4)
            tk.Frame(card, bg=clr, height=3).pack(fill="x")
            tk.Label(card, text=sym, font=(FONT_MAIN, 9, "bold"),
                     bg=COLORS["card"], fg=clr).pack(pady=(6, 0))
            tk.Label(card, text=str(val), font=(FONT_MAIN, 17, "bold"),
                     bg=COLORS["card"], fg=clr).pack()
            tk.Label(card, text=name, font=(FONT_MAIN, 8),
                     bg=COLORS["card"], fg=COLORS["subtext"]).pack(pady=(0, 8))

        hist_card = make_card(self._stats_frame)
        hist_card.pack(fill="both", expand=True)

        hdr = tk.Frame(hist_card, bg=COLORS["card2"])
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"  {poll} Frequency Distribution",
                 font=(FONT_MAIN, 10, "bold"),
                 bg=COLORS["card2"], fg=COLORS["text"]).pack(side="left", pady=8)
        tk.Label(hdr, text=f"  Safe Limit: {limit} µg/m³  ",
                 font=(FONT_MAIN, 9),
                 bg=COLORS["card2"], fg=COLORS["subtext"]).pack(side="right", pady=8)

        fig = Figure(figsize=(12, 3.6), dpi=96)
        ax  = fig.add_subplot(111)
        n, bins, patches = ax.hist(series.dropna(), bins=45,
                                   edgecolor=COLORS["bg"], alpha=0.8, linewidth=0.4,
                                   color=COLORS["accent2"])
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(COLORS["bad"] if left > limit else COLORS["accent2"])
            patch.set_alpha(0.75)

        ax.axvline(stats["mean"],   color=COLORS["accent"],
                   linestyle="--", linewidth=1.8, label=f"Mean  {stats['mean']}")
        ax.axvline(stats["median"], color=COLORS["good"],
                   linestyle=":",  linewidth=1.8, label=f"Median  {stats['median']}")
        ax.axvline(limit,           color=COLORS["bad"],
                   linestyle="-",  linewidth=2,   label=f"Safe Limit  {limit}")
        ax.set_xlabel(f"{poll} Average (µg/m³)", labelpad=8)
        ax.set_ylabel("Frequency", labelpad=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["border2"],
                  labelcolor=COLORS["text"], fontsize=9)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=hist_card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # ── TAB 5 : CRUD ─────────────────────────
    def _tab_crud(self):
        tab = tk.Frame(self.nb, bg=COLORS["bg"])
        self.nb.add(tab, text="  🗄️  MongoDB CRUD  ")

        # Status pill
        status_bar = tk.Frame(tab, bg=COLORS["bg"])
        status_bar.pack(fill="x", padx=20, pady=(14, 4))
        pill_clr = COLORS["good"] if self.db.connected else COLORS["bad"]
        pill_bg  = "#0d2e1a" if self.db.connected else "#2e0d0d"
        pill = tk.Frame(status_bar, bg=pill_bg,
                        highlightbackground=pill_clr, highlightthickness=1)
        pill.pack(side="left")
        tk.Label(pill,
                 text=f"  ● MongoDB {'Connected' if self.db.connected else 'Offline'}  ",
                 font=(FONT_MAIN, 10, "bold"),
                 bg=pill_bg, fg=pill_clr).pack(pady=4, padx=6)
        tk.Label(status_bar,
                 text="  All operations run on MongoDB. Falls back to CSV if offline.",
                 font=(FONT_MAIN, 9),
                 bg=COLORS["bg"], fg=COLORS["subtext"]).pack(side="left", padx=10)

        # CRUD buttons
        btn_row = tk.Frame(tab, bg=COLORS["bg"])
        btn_row.pack(fill="x", padx=20, pady=8)

        for label, cmd, clr, bg in [
            ("📖  READ",    self._crud_read,   COLORS["accent"],  "#0d2030"),
            ("✏️   UPDATE",  self._crud_update, COLORS["accent2"], "#18102e"),
            ("🗑️   DELETE",  self._crud_delete, COLORS["bad"],     "#2e0d0d"),
            ("📊  COUNT",   self._crud_count,  COLORS["good"],    "#0d2e1a"),
        ]:
            f = tk.Frame(btn_row, bg=bg,
                         highlightbackground=clr, highlightthickness=1)
            f.pack(side="left", padx=6)
            tk.Button(f, text=label, command=cmd,
                      bg=bg, fg=clr,
                      font=(FONT_MAIN, 10, "bold"), relief="flat",
                      cursor="hand2", padx=18, pady=9,
                      activebackground=COLORS["highlight"],
                      activeforeground=clr).pack()

        # Log
        log_card = make_card(tab)
        log_card.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        log_hdr = tk.Frame(log_card, bg=COLORS["card2"])
        log_hdr.pack(fill="x")
        tk.Label(log_hdr, text="  Operation Log",
                 font=(FONT_MAIN, 10, "bold"),
                 bg=COLORS["card2"], fg=COLORS["accent"]).pack(side="left", pady=8)
        tk.Button(log_hdr, text="Clear  ✕",
                  command=lambda: (self._crud_log.config(state="normal"),
                                   self._crud_log.delete("1.0", "end"),
                                   self._crud_log.config(state="disabled")),
                  bg=COLORS["card2"], fg=COLORS["subtext"],
                  font=(FONT_MAIN, 8), relief="flat",
                  cursor="hand2", padx=8).pack(side="right", pady=4, padx=8)

        self._crud_log = tk.Text(log_card, bg=COLORS["bg2"], fg=COLORS["text"],
                                  font=(FONT_MONO, 10), relief="flat",
                                  insertbackground=COLORS["text"],
                                  wrap="word", state="disabled")
        sb = ttk.Scrollbar(log_card, command=self._crud_log.yview)
        self._crud_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._crud_log.pack(fill="both", expand=True, padx=2, pady=2)

        self._crud_log.tag_config("ok",   foreground=COLORS["good"])
        self._crud_log.tag_config("err",  foreground=COLORS["bad"])
        self._crud_log.tag_config("info", foreground=COLORS["accent"])
        self._crud_log.tag_config("head", foreground=COLORS["moderate"])
        self._crud_log.tag_config("dim",  foreground=COLORS["subtext"])

        self._log("═" * 50 + "\n  MongoDB CRUD Console\n" + "═" * 50, "head")
        if not self.db.connected:
            self._log("MongoDB not detected. Start MongoDB and relaunch to enable DB ops.", "err")
        else:
            self._log(f"Database   : {DB_NAME}", "info")
            self._log(f"Collection : {COLLECTION_NAME}", "info")
            self._log(f"Documents  : {self.db.count()} loaded", "ok")

    def _log(self, msg: str, tag: str = "info"):
        self._crud_log.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._crud_log.insert("end", f"\n[{ts}]  {msg}", tag)
        self._crud_log.see("end")
        self._crud_log.config(state="disabled")

    def _crud_read(self):
        if not self.db.connected:
            self._log("MongoDB offline — showing CSV sample instead.", "err")
            df = self.dp.filter()
            self._log(f"CSV records: {len(df):,}  |  Sample (5 rows):", "dim")
            self._log(df.head(5)[["city","pollutant_id","pollutant_avg"]].to_string(), "info")
            return
        poll = simpledialog.askstring("READ",
                                      "Enter pollutant to query\n(e.g. PM2.5, NO2, SO2)",
                                      parent=self, initialvalue="PM2.5")
        if not poll:
            return
        docs = self.db.find({"pollutant_id": poll.strip().upper()}, limit=8)
        if docs:
            self._log(f"READ  ›  pollutant_id = {poll.upper()}  ({len(docs)} docs)", "head")
            for d in docs:
                self._log(f"  {d.get('city','?'):<18} avg={d.get('pollutant_avg','?'):<8} "
                           f"| {d.get('station','?')[:42]}", "ok")
        else:
            self._log(f"No records found for pollutant: {poll}", "err")

    def _crud_update(self):
        if not self.db.connected:
            self._log("MongoDB offline — UPDATE unavailable.", "err")
            return
        station = simpledialog.askstring("UPDATE", "Enter exact station name:", parent=self)
        if not station:
            return
        poll = simpledialog.askstring("UPDATE", "Enter pollutant (e.g. PM2.5):",
                                      parent=self, initialvalue="PM2.5")
        if not poll:
            return
        new_val = simpledialog.askfloat("UPDATE", "Enter new pollutant_avg value:", parent=self)
        if new_val is None:
            return
        ok = self.db.update_record(station.strip(), poll.strip().upper(), new_val)
        if ok:
            self._log(f"UPDATE ✓  station='{station}'  {poll} → {new_val}", "ok")
        else:
            self._log(f"UPDATE ✗  No match for station='{station}', pollutant='{poll}'", "err")

    def _crud_delete(self):
        if not self.db.connected:
            self._log("MongoDB offline — DELETE unavailable.", "err")
            return
        city = simpledialog.askstring("DELETE",
                                      "Enter city name to delete all records:", parent=self)
        if not city:
            return
        if messagebox.askyesno("Confirm Delete",
                                f"Delete ALL records for city: '{city}'?\nThis cannot be undone."):
            deleted = self.db.delete_city(city.strip())
            if deleted:
                self._log(f"DELETE ✓  {deleted} document(s) removed for city='{city}'", "ok")
            else:
                self._log(f"DELETE ✗  City '{city}' not found in DB.", "err")

    def _crud_count(self):
        if not self.db.connected:
            self._log(f"MongoDB offline. CSV has {len(self.dp.df):,} records.", "err")
            return
        count = self.db.count()
        self._log(f"COUNT  ›  '{COLLECTION_NAME}'  →  {count:,} documents", "ok")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = AirQualityApp()
    app.mainloop()