import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ALGORITHM FUNCTIONS
# =====================

# Greedy Fractional Algorithm
def greedy_fractional(stocks, budget):
    stocks_sorted = sorted(stocks, key=lambda x: x[2]/x[1], reverse=True)
    total_cost = 0
    result = []
    for name, price, ret in stocks_sorted:
        if total_cost + price <= budget:
            result.append((name, price, ret))
            total_cost += price
        else:
            fraction = (budget - total_cost)/price
            if fraction > 0:
                result.append((name, price*fraction, ret*fraction))
                total_cost += price*fraction
            break
    total_return = sum(r[2] for r in result)
    return result, total_cost, total_return


# Dynamic Programming (0/1 Knapsack)
def dp_knapsack(stocks, budget):
    n = len(stocks)
    W = int(budget)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            if stocks[i-1][1] <= w:
                dp[i][w] = max(dp[i-1][w],
                               dp[i-1][w-int(stocks[i-1][1])] + stocks[i-1][2])
            else:
                dp[i][w] = dp[i-1][w]

    # Traceback
    w = W
    selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(stocks[i-1])
            w -= int(stocks[i-1][1])

    total_cost = sum(s[1] for s in selected)
    total_return = sum(s[2] for s in selected)
    return selected[::-1], total_cost, total_return


# Backtracking (Exhaustive Search)
def backtracking_selection(stocks, budget):
    n = len(stocks)
    best_return = 0
    best_combo = []

    def backtrack(i, current_combo, current_cost, current_return):
        nonlocal best_return, best_combo

        if i == n:
            if current_return > best_return:
                best_return = current_return
                best_combo = current_combo[:]
            return

        # Prune if cost exceeds budget
        if current_cost > budget:
            return

        # Include stock[i]
        name, price, ret = stocks[i]
        if current_cost + price <= budget:
            current_combo.append(stocks[i])
            backtrack(i + 1, current_combo, current_cost + price, current_return + ret)
            current_combo.pop()

        # Exclude stock[i]
        backtrack(i + 1, current_combo, current_cost, current_return)

    backtrack(0, [], 0, 0)
    total_cost = sum(s[1] for s in best_combo)
    total_return = sum(s[2] for s in best_combo)
    return best_combo, total_cost, total_return


# =====================
# STREAMLIT INTERFACE
# =====================

st.set_page_config(page_title="Hybrid Stock Portfolio Optimizer", layout="wide")
st.title("📈 Dynamic Stock Portfolio Rebalancer (Greedy + DP + Backtracking)")

st.markdown("""
Upload a CSV file with **Stock**, **Price**, and **ExpectedReturn** columns.  
The system automatically finds the best combination using:
- ⚡ Greedy Fractional Algorithm  
- 🧠 Dynamic Programming (0/1 Knapsack)  
- 🔍 Backtracking Exhaustive Search  
""")

uploaded_file = st.file_uploader("📁 Upload your stock dataset (CSV file)", type=["csv"])
budget = st.number_input("💰 Enter your total investment budget ($):", value=5000, min_value=100)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Stock", "Price", "ExpectedReturn"}

    if not required_columns.issubset(df.columns):
        st.error("❌ CSV must contain 'Stock', 'Price', and 'ExpectedReturn' columns.")
    else:
        stocks = [tuple(x) for x in df[["Stock", "Price", "ExpectedReturn"]].values]
        st.write("### ✅ Loaded Stocks:")
        st.dataframe(df)

        if st.button("🚀 Run Hybrid Optimization"):
            # Greedy Algorithm
            greedy_sel, greedy_cost, greedy_ret = greedy_fractional(stocks, budget)
            st.subheader("⚡ Greedy (Fractional) Solution")
            st.dataframe(pd.DataFrame(greedy_sel, columns=["Stock", "Invested ($)", "Expected Return"]))
            st.info(f"**Total Cost:** ${greedy_cost:.2f} | **Total Expected Return:** {greedy_ret:.2f}")

            # Dynamic Programming
            dp_sel, dp_cost, dp_ret = dp_knapsack(stocks, budget)
            st.subheader("🧠 Dynamic Programming (0/1 Knapsack) Solution")
            st.dataframe(pd.DataFrame(dp_sel, columns=["Stock", "Invested ($)", "Expected Return"]))
            st.info(f"**Total Cost:** ${dp_cost:.2f} | **Total Expected Return:** {dp_ret:.2f}")

            # Backtracking (Exhaustive Search)
            bt_sel, bt_cost, bt_ret = backtracking_selection(stocks, budget)
            st.subheader("🔍 Backtracking (Optimal) Solution")
            st.dataframe(pd.DataFrame(bt_sel, columns=["Stock", "Invested ($)", "Expected Return"]))
            st.success(f"**Optimal Total Cost:** ${bt_cost:.2f} | **Optimal Total Return:** {bt_ret:.2f}")

            # Comparison Summary
            st.markdown("---")
            st.subheader("📊 Comparison Summary")
            comparison_df = pd.DataFrame({
                "Algorithm": ["Greedy", "Dynamic Programming", "Backtracking (Optimal)"],
                "Total Cost ($)": [round(greedy_cost, 2), round(dp_cost, 2), round(bt_cost, 2)],
                "Total Return": [round(greedy_ret, 2), round(dp_ret, 2), round(bt_ret, 2)]
            })
            st.dataframe(comparison_df)

            # Chart
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(comparison_df["Algorithm"], comparison_df["Total Return"])
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Total Expected Return")
            ax.set_title("Algorithm Performance Comparison")
            st.pyplot(fig)
else:
    st.info("📥 Please upload a CSV file to begin.")
