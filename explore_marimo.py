import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    return mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Marimo Playground — Reactivity & Visualisation

    Three self-contained sections to get a feel for Marimo's reactive flow:

    1. **Property score calculator** — sliders that feed a live formula
    2. **Fake rental dataset** — filterable table driven by UI controls
    3. **Score distribution chart** — updates as you change the dataset size
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Property score calculator
    """)
    return


@app.cell
def _(mo):
    location_score = mo.ui.slider(1, 10, value=7, label="Location score")
    location_score
    return (location_score,)


@app.cell
def _(mo):
    amenities_score = mo.ui.slider(1, 10, value=5, label="Amenities score")
    amenities_score
    return (amenities_score,)


@app.cell
def _(mo):
    price_per_night = mo.ui.number(50, 1000, value=200, step=10, label="Price per night (€)")
    price_per_night
    return (price_per_night,)


@app.cell
def _(amenities_score, location_score, mo, price_per_night):
    value_ratio = (location_score.value * 0.5 + amenities_score.value * 0.5) / (price_per_night.value / 100)
    overall = round(min(value_ratio * 10, 10), 2)

    mo.md(
        f"""
        ### Result

        | Metric | Value |
        |--------|-------|
        | Weighted quality score | **{(location_score.value * 0.5 + amenities_score.value * 0.5):.1f} / 10** |
        | Value for money index | **{value_ratio:.2f}** |
        | Overall rating | **{overall} / 10** |

        > Move any slider above — this cell updates instantly without pressing anything.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Fake rental dataset
    """)
    return


@app.cell
def _(mo):
    n_properties = mo.ui.slider(10, 200, value=50, step=10, label="Number of properties to generate")
    n_properties
    return (n_properties,)


@app.cell
def _(mo, n_properties, np, pd):
    rng = np.random.default_rng(42)
    property_types = ["villa", "apartment", "cottage", "chalet", "beach house"]

    df = pd.DataFrame({
        "name": [f"Property {i+1}" for i in range(n_properties.value)],
        "type": rng.choice(property_types, n_properties.value),
        "bedrooms": rng.integers(1, 7, n_properties.value),
        "price_per_night": rng.integers(60, 900, n_properties.value),
        "location_score": rng.integers(1, 11, n_properties.value),
        "amenities_score": rng.integers(1, 11, n_properties.value),
    })

    min_price = mo.ui.slider(
        int(df.price_per_night.min()),
        int(df.price_per_night.max()),
        value=int(df.price_per_night.min()),
        label="Min price per night (€)",
    )
    min_price
    return df, min_price


@app.cell
def _(mo):
    selected_type = mo.ui.dropdown(
        ["all", "villa", "apartment", "cottage", "chalet", "beach house"],
        value="all",
        label="Property type filter",
    )
    selected_type
    return (selected_type,)


@app.cell
def _(df, min_price, mo, selected_type):
    filtered = df[df.price_per_night >= min_price.value]
    if selected_type.value != "all":
        filtered = filtered[filtered.type == selected_type.value]

    mo.md(f"**{len(filtered)} properties** match your filters")
    return (filtered,)


@app.cell
def _(filtered, mo):
    mo.ui.table(filtered.reset_index(drop=True))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Score distribution chart
    """)
    return


@app.cell
def _(filtered, mo, np):
    scores = (filtered.location_score * 0.5 + filtered.amenities_score * 0.5).values

    bins = np.arange(0.5, 11.5, 1)
    counts, edges = np.histogram(scores, bins=bins)
    labels = [str(int(e + 0.5)) for e in edges[:-1]]

    bar_width = 30
    chart_lines = ["```", "Score  Count  Bar"]
    for label, count in zip(labels, counts):
        bar = "█" * int(count * bar_width / max(counts, default=1))
        chart_lines.append(f"  {label:>5}  {count:>5}  {bar}")
    chart_lines.append("```")

    mo.md(
        f"""
        ### Quality score distribution (n={len(filtered)})

        {chr(10).join(chart_lines)}

        Mean: **{scores.mean():.2f}** · Std: **{scores.std():.2f}**
        """
    )
    return


if __name__ == "__main__":
    app.run()
