import os

# Store Matplotlib cache in a writable local folder for consistent script runs.
os.makedirs(".matplotlib", exist_ok=True)
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Use a non-interactive backend so the script runs cleanly in headless setups.
plt.switch_backend("Agg")


def generate_dataset(num_rows=150, seed=42):
    """Generate a synthetic sales dataset for analysis."""
    rng = np.random.default_rng(seed)

    start_date = np.datetime64("2025-01-01")
    end_date = np.datetime64("2025-04-30")
    date_range_days = (end_date - start_date).astype(int)

    data = {
        "Date": start_date + rng.integers(0, date_range_days + 1, size=num_rows).astype(
            "timedelta64[D]"
        ),
        "Product": rng.choice(list("ABCDE"), size=num_rows),
        "Region": rng.choice(["North", "South", "East", "West"], size=num_rows),
        "Sales": rng.integers(100, 1000, size=num_rows),
    }

    df = pd.DataFrame(data)
    return df


def run_eda(df):
    """Perform exploratory data analysis tasks."""
    print("\n" + "=" * 60)
    print("SECTION 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # Convert Date to datetime and set it as the index for time-series analysis.
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Weekly resampling to calculate total weekly sales.
    weekly_sales = df["Sales"].resample("W").sum()
    print("\nWeekly Total Sales:")
    print(weekly_sales)

    # Pivot table summarizing sales by product.
    pivot_table = df.pivot_table(
        index="Product",
        values="Sales",
        aggfunc=["sum", "mean"],
    )
    print("\nPivot Table: Sales by Product (Sum and Mean)")
    print(pivot_table)

    # Correlation heatmap using numeric columns only.
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap_day5.png", dpi=300)
    plt.close()

    return df


def run_dashboard(df):
    """Create dashboard subplots for product and daily sales trends."""
    print("\n" + "=" * 60)
    print("SECTION 2: DASHBOARD")
    print("=" * 60)

    total_sales_by_product = (
        df.groupby("Product", as_index=False)["Sales"].sum().sort_values("Product")
    )
    daily_sales = df["Sales"].resample("D").sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=total_sales_by_product, x="Product", y="Sales", ax=axes[0], color="steelblue")
    axes[0].set_title("Total Sales by Product")
    axes[0].set_xlabel("Product")
    axes[0].set_ylabel("Total Sales")

    sns.lineplot(data=daily_sales, x="Date", y="Sales", ax=axes[1], marker="o", color="darkgreen")
    axes[1].set_title("Daily Sales Trend")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Daily Sales")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("dashboard_day5.png", dpi=300)
    plt.close()


def run_scenario_analysis(df):
    """Perform grouped scenario analysis and Pareto analysis."""
    print("\n" + "=" * 60)
    print("SECTION 3: SCENARIO ANALYSIS")
    print("=" * 60)

    # Multi-level groupby by Product and Date with renamed aggregation columns.
    grouped_sales = (
        df.groupby(["Product", pd.Grouper(freq="D")])["Sales"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Total_Sales", "count": "Sales_Count"})
    )
    print("\nGrouped Sales by Product and Date:")
    print(grouped_sales.head())

    # Scenario revenue projection assuming a 15% uplift.
    df["Projected_Revenue"] = df["Sales"] * 1.15

    # Pareto analysis for top products by total sales.
    product_totals = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    top_5_products = product_totals.head(5)
    pareto_df = top_5_products.reset_index()
    pareto_df.columns = ["Product", "Total_Sales"]
    pareto_df["Percentage_Contribution"] = (
        pareto_df["Total_Sales"] / product_totals.sum() * 100
    ).round(2)

    print("\nPareto Analysis: Top 5 Products by Total Sales")
    print(pareto_df)

    return df


def run_extra_analysis(df):
    """Create extra analysis visualizations."""
    print("\n" + "=" * 60)
    print("SECTION 4: EXTRA ANALYSIS")
    print("=" * 60)

    # Boxplot for sales distribution by product.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df.reset_index(), x="Product", y="Sales", color="lightseagreen")
    plt.title("Sales Distribution by Product")
    plt.xlabel("Product")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig("boxplot_day5.png", dpi=300)
    plt.close()

    # Pairplot using numeric columns only.
    pairplot_df = df.select_dtypes(include=[np.number]).copy()
    pairplot_grid = sns.pairplot(pairplot_df)
    pairplot_grid.fig.savefig("pairplot_day5.png", dpi=300)
    plt.close(pairplot_grid.fig)

    # Cumulative sales plot based on daily aggregated sales.
    daily_sales = df["Sales"].resample("D").sum()
    cumulative_sales = daily_sales.cumsum()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=cumulative_sales.index, y=cumulative_sales.values, color="crimson")
    plt.title("Cumulative Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cumsum_day5.png", dpi=300)
    plt.close()


def main():
    """Run the full Day 5 data analyst workflow."""
    sns.set_theme(style="whitegrid")

    # Generate the synthetic dataset and show a preview in the terminal.
    df = generate_dataset(num_rows=150, seed=42)
    print("Synthetic Dataset Preview:")
    print(df.head())
    print(f"\nTotal Rows Generated: {len(df)}")

    df = run_eda(df)
    run_dashboard(df)
    df = run_scenario_analysis(df)
    run_extra_analysis(df)

    print("\nFiles saved successfully:")
    print("- heatmap_day5.png")
    print("- dashboard_day5.png")
    print("- boxplot_day5.png")
    print("- pairplot_day5.png")
    print("- cumsum_day5.png")


if __name__ == "__main__":
    main()
