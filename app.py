import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(
    page_title="Life Expectancy Across Hispanic Countries",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 Life Expectancy Across Hispanic Countries")
st.caption("Created by Charles")
st.markdown(
    """
Explore historical life expectancy at birth in Spanish-speaking countries and
Equatorial Guinea, fit a polynomial regression model (degree 3 or higher), and
analyze the model using real-world context. Data are retrieved from the **World
Bank World Development Indicators API** when the app runs.
"""
)

st.info(
    "Project note: this app provides a reproducible starting point for your "
    "Function Analysis Project. Check your teacher's country/entity reservation "
    "sheet before choosing a country for your final presentation."
)

# World Bank country codes for Spanish-speaking countries and Equatorial Guinea.
COUNTRIES = {
    "Argentina": "ARG",
    "Bolivia": "BOL",
    "Chile": "CHL",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Cuba": "CUB",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Mexico": "MEX",
    "Nicaragua": "NIC",
    "Panama": "PAN",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Spain": "ESP",
    "Uruguay": "URY",
    "Venezuela": "VEN",
}
INDICATOR = "SP.DYN.LE00.IN"
API_BASE = "https://api.worldbank.org/v2"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_country_data(country_code: str) -> pd.DataFrame:
    """Download all available annual life expectancy data for one country."""
    url = f"{API_BASE}/country/{country_code}/indicator/{INDICATOR}"
    params = {"format": "json", "per_page": 2000}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return pd.DataFrame(columns=["Year", "Life expectancy (years)"])

    records = []
    for item in payload[1]:
        value = item.get("value")
        year = item.get("date")
        if value is not None and year is not None:
            records.append({"Year": int(year), "Life expectancy (years)": float(value)})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("Year").drop_duplicates("Year").reset_index(drop=True)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_all_countries() -> pd.DataFrame:
    """Download available observations for all listed countries."""
    frames = []
    for name, code in COUNTRIES.items():
        try:
            part = fetch_country_data(code)
            if not part.empty:
                part["Country"] = name
                frames.append(part)
        except Exception:
            # Keep the app usable if one country's endpoint is temporarily unavailable.
            continue
    if not frames:
        return pd.DataFrame(columns=["Year", "Life expectancy (years)", "Country"])
    return pd.concat(frames, ignore_index=True)


def format_equation(coef, center_year, degree):
    """Format a polynomial using t = year - center_year for numerical stability."""
    terms = []
    powers = list(range(degree, -1, -1))
    for c, power in zip(coef, powers):
        if abs(c) < 1e-10:
            continue
        if power == 0:
            term = f"{c:.6g}"
        elif power == 1:
            term = f"{c:.6g}(t)"
        else:
            term = f"{c:.6g}(t^{power})"
        terms.append(term)
    equation = " + ".join(terms).replace("+ -", "- ")
    return f"LE(t) = {equation}, where t = year − {center_year}"


def real_roots_in_interval(poly, lower, upper, tolerance=1e-7):
    roots = np.roots(poly)
    real = []
    for root in roots:
        if abs(np.imag(root)) < tolerance:
            x = float(np.real(root))
            if lower <= x <= upper:
                real.append(x)
    return sorted(real)


def year_text(year):
    return str(int(round(year)))


def analyze_model(coef, degree, center_year, first_year, last_year):
    """Return model analysis in calendar-year coordinates."""
    p = np.poly1d(coef)
    dp = np.polyder(p)
    ddp = np.polyder(dp)
    lo = first_year - center_year
    hi = last_year - center_year

    # Relative extrema occur where the first derivative is zero.
    critical = real_roots_in_interval(dp, lo, hi)
    extrema = []
    for t in critical:
        second = float(ddp(t)) if degree >= 2 else 0.0
        kind = "local maximum" if second < -1e-8 else (
            "local minimum" if second > 1e-8 else "stationary point"
        )
        extrema.append((center_year + t, float(p(t)), kind))

    # Fastest increase/decrease on the observed interval: evaluate derivative
    # at endpoints and at roots of the second derivative.
    rate_candidates = [lo, hi]
    rate_candidates += real_roots_in_interval(ddp, lo, hi) if degree >= 2 else []
    rates = [(center_year + t, float(dp(t))) for t in rate_candidates]
    fastest_increase = max(rates, key=lambda pair: pair[1])
    fastest_decrease = min(rates, key=lambda pair: pair[1])

    zeros = real_roots_in_interval(p, lo, hi)
    return {
        "poly": p,
        "derivative": dp,
        "extrema": extrema,
        "fastest_increase": fastest_increase,
        "fastest_decrease": fastest_decrease,
        "zeros": [center_year + z for z in zeros],
    }


with st.sidebar:
    st.header("Model settings")
    selected_country = st.selectbox("Choose a country", list(COUNTRIES.keys()), index=11)
    degree = st.slider("Regression degree", min_value=3, max_value=5, value=3, step=1)
    st.caption("Degree 3 is cubic. Higher-degree models may fit historical data more closely but can behave unrealistically outside the data range.")
    future_year = st.number_input(
        "Future year for extrapolation",
        min_value=2025,
        max_value=2150,
        value=min(datetime.now().year + 10, 2150),
        step=1,
    )
    show_other_countries = st.checkbox("Show country comparison chart", value=True)

try:
    with st.spinner("Loading historical data from the World Bank..."):
        data = fetch_country_data(COUNTRIES[selected_country])
except Exception as exc:
    st.error(
        "The app could not download the World Bank data. Check your internet "
        "connection and try again. Details: " + str(exc)
    )
    st.stop()

if data.empty:
    st.warning("No observations were returned for this country. Choose another country.")
    st.stop()

# Keep the fit stable and make the observed years explicit.
first_year = int(data["Year"].min())
last_year = int(data["Year"].max())
center_year = int(round(data["Year"].mean()))
x = data["Year"].to_numpy(dtype=float) - center_year
y = data["Life expectancy (years)"].to_numpy(dtype=float)
actual_degree = min(degree, len(data) - 1)
if actual_degree < 3:
    st.error("There are too few observations to fit a degree-3 model for this country.")
    st.stop()

coef = np.polyfit(x, y, actual_degree)
analysis = analyze_model(coef, actual_degree, center_year, first_year, last_year)
poly = analysis["poly"]
model_y = poly(x)
r_squared = 1 - (np.sum((y - model_y) ** 2) / np.sum((y - np.mean(y)) ** 2))

latest = data.iloc[-1]
earliest = data.iloc[0]
predicted_future = float(poly(float(future_year) - center_year))
predicted_5_year = float(poly(float(last_year + 5) - center_year))

st.subheader(f"{selected_country}: life expectancy at birth")
st.caption(
    f"Historical observations available: {first_year}–{last_year}. "
    "Life expectancy is measured in years and represents the expected lifespan "
    "of a newborn under the mortality patterns of that year."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Earliest available value", f"{earliest['Life expectancy (years)']:.2f} years", year_text(earliest["Year"]))
m2.metric("Latest available value", f"{latest['Life expectancy (years)']:.2f} years", year_text(latest["Year"]))
m3.metric("Overall change", f"{latest['Life expectancy (years)'] - earliest['Life expectancy (years)']:+.2f} years")
m4.metric("Model R²", f"{r_squared:.4f}")

# Chart with actual points and fitted curve.
plot_years = np.linspace(first_year, last_year, 500)
plot_values = poly(plot_years - center_year)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data["Year"], y=data["Life expectancy (years)"],
    mode="markers", name="Historical data",
    marker=dict(size=7), hovertemplate="Year %{x}<br>Life expectancy %{y:.2f} years<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=plot_years, y=plot_values,
    mode="lines", name=f"Degree-{actual_degree} regression",
    hovertemplate="Year %{x:.0f}<br>Model %{y:.2f} years<extra></extra>"
))
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Life expectancy at birth (years)",
    legend_title="",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("1. Raw data table")
st.write("Inputs are calendar years (x); outputs are life expectancy at birth in years (y).")
display_data = data.rename(columns={"Year": "Input: Year (x)", "Life expectancy (years)": "Output: Life expectancy (y, years)"})
st.dataframe(display_data, use_container_width=True, hide_index=True)
csv = display_data.to_csv(index=False).encode("utf-8")
st.download_button("Download this country's raw data as CSV", csv, f"{selected_country.lower().replace(' ', '_')}_life_expectancy.csv", "text/csv")

st.subheader("2. Regression equation")
st.markdown(f"**Model:** `{format_equation(coef, center_year, actual_degree)}`")
st.markdown(f"**Goodness of fit:** \(R^2 = {r_squared:.4f}\). An R² closer to 1 means the model follows the historical observations more closely.")
st.caption("The equation uses t = year − the center year shown above, instead of raw calendar years, to improve numerical stability. This is mathematically equivalent to a polynomial in the year.")

st.subheader("3. Function analysis")
st.markdown(f"**Observed domain:** {first_year} ≤ year ≤ {last_year}.")
st.markdown(
    f"**Observed-data range:** {data['Life expectancy (years)'].min():.2f} to "
    f"{data['Life expectancy (years)'].max():.2f} years. "
    "The model's range over the observed interval can differ slightly from the raw-data range."
)

if analysis["extrema"]:
    st.markdown("**Local relative maxima/minima predicted by the model (within the observed years):**")
    for yr, val, kind in analysis["extrema"]:
        st.write(f"- The model predicts a {kind} around {yr:.2f} (approximately {year_text(yr)}), at about {val:.2f} years of life expectancy.")
else:
    st.write("The fitted model has no local maximum or minimum inside the observed time interval.")

inc_year, inc_rate = analysis["fastest_increase"]
dec_year, dec_rate = analysis["fastest_decrease"]
st.markdown(
    f"**Increasing fastest:** around {inc_year:.2f} ({year_text(inc_year)}), "
    f"at {inc_rate:+.4f} life-expectancy years per calendar year."
)
st.markdown(
    f"**Decreasing fastest:** around {dec_year:.2f} ({year_text(dec_year)}), "
    f"at {dec_rate:+.4f} life-expectancy years per calendar year."
)

if analysis["zeros"]:
    st.markdown("**Zeros within the observed interval:** " + ", ".join(year_text(z) for z in analysis["zeros"]) + ".")
else:
    st.markdown("**Zeros:** The model has no zero within the observed historical interval. A life expectancy of zero years is not a realistic outcome for this context.")

# Explain intervals by checking model derivative sign at each interval midpoint.
critical_years = [yr for yr, _, _ in analysis["extrema"]]
boundaries = [float(first_year)] + sorted(critical_years) + [float(last_year)]
interval_text = []
for left, right in zip(boundaries[:-1], boundaries[1:]):
    midpoint = ((left + right) / 2) - center_year
    slope = float(analysis["derivative"](midpoint))
    trend = "increasing" if slope > 1e-6 else ("decreasing" if slope < -1e-6 else "approximately flat")
    interval_text.append(f"- From about {left:.1f} to {right:.1f}, the model is **{trend}**.")
st.markdown("**Increasing/decreasing intervals:**")
st.markdown("\n".join(interval_text) if interval_text else "Not enough data to determine intervals.")

st.subheader("4. Extrapolation: a future prediction")
st.markdown(
    f"According to the degree-{actual_degree} regression model, the predicted life expectancy "
    f"for **{selected_country} in {int(future_year)}** is **{predicted_future:.2f} years**."
)
if future_year > last_year:
    st.warning(
        f"This is extrapolation because {int(future_year)} is beyond the last available observation ({last_year}). "
        "Polynomial models can curve sharply outside the data range, so treat this as a mathematical prediction—not an official forecast."
    )
else:
    st.info("The selected year is not beyond the historical data, so this value is interpolation rather than extrapolation.")
st.write(
    f"For an additional reference, the model predicts {predicted_5_year:.2f} years in {last_year + 5}, "
    "five years after the last available observation."
)

st.subheader("5. Possible explanations for major changes")
st.markdown(
    """
Use this section to connect the graph to real history. Do not claim a cause without checking a reliable source.
Possible factors that can affect national life expectancy include:
- access to doctors, hospitals, vaccination, and public-health programs;
- changes in infant and maternal mortality;
- economic growth, poverty, nutrition, and clean drinking water;
- epidemics, conflict, natural disasters, and disruptions to health services;
- aging populations and changes in heart disease, cancer, and other causes of death.
"""
)
st.text_area(
    "Write your evidence-based explanation here",
    placeholder="Example: Life expectancy rose during this period. A possible explanation is improved access to healthcare, but I would verify this using a trustworthy historical or public-health source...",
    height=110,
)

if show_other_countries:
    st.subheader("6. Compare countries")
    try:
        with st.spinner("Loading comparison data..."):
            all_data = fetch_all_countries()
        if all_data.empty:
            st.warning("Comparison data could not be loaded.")
        else:
            comparison = go.Figure()
            for country, group in all_data.groupby("Country"):
                comparison.add_trace(go.Scatter(
                    x=group["Year"], y=group["Life expectancy (years)"],
                    mode="lines", name=country,
                    opacity=1.0 if country == selected_country else 0.65,
                ))
            comparison.update_layout(
                xaxis_title="Year",
                yaxis_title="Life expectancy at birth (years)",
                legend_title="Country",
                hovermode="x unified",
            )
            st.plotly_chart(comparison, use_container_width=True)
            st.caption("The comparison chart shows historical observations, not regression curves, for the listed countries.")
    except Exception:
        st.warning("The country comparison could not be loaded right now. The selected-country analysis above is still available.")

st.subheader("Data source and citations")
st.markdown(
    f"""
- **World Bank indicator:** Life expectancy at birth, total (years), indicator `{INDICATOR}`.
- **Indicator page:** https://data.worldbank.org/indicator/{INDICATOR}
- **API endpoint pattern:** https://api.worldbank.org/v2/country/{COUNTRIES[selected_country]}/indicator/{INDICATOR}?format=json&per_page=2000
- Data are fetched when the app runs and may be revised by the World Bank. Record the access date in your school presentation.
"""
)

st.subheader("Presentation checklist")
st.markdown(
    """
- [ ] Confirm your country/entity is not already reserved by another student.
- [ ] Include the raw-data table with labeled x and y values.
- [ ] Include the scatter plot and regression curve.
- [ ] Write the regression equation and explain what x and y mean.
- [ ] Explain local extrema, increasing/decreasing intervals, fastest rate of change, domain, range, and zeros where meaningful.
- [ ] Research a historical reason for at least one major change.
- [ ] Include the future prediction and clearly label it as extrapolation.
- [ ] Mention that the polynomial is a model, not an official demographic forecast.
"""
)

st.caption("Built with Streamlit, pandas, NumPy, Plotly, and the World Bank API.")
