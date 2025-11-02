import streamlit as st
import pandas as pd
import numpy as np
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


def dp_knapsack(stocks, budget):
    n = len(stocks)
    W = int(budget)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            if stocks[i-1][1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-int(stocks[i-1][1])] + stocks[i-1][2])
            else:
                dp[i][w] = dp[i-1][w]
    # Traceback
    w = W
    selected = []
    for i in range(n,0,-1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(stocks[i-1])
            w -= int(stocks[i-1][1])
    total_cost = sum(s[1] for s in selected)
    total_return = sum(s[2] for s in selected)
    return selected[::-1], total_cost, total_return


def backtracking_selection(stocks, budget, greedy_ret, dp_ret):
    n = len(stocks)
    best_return = max(greedy_ret, dp_ret)
    best_combo = []

    stocks_sorted = sorted(stocks, key=lambda x: x[2]/x[1], reverse=True)

    def backtrack(i, combo, cost, ret):
        nonlocal best_return, best_combo
        if cost > budget:
            return
        # Upper bound using fractional estimation
        remaining_budget = budget - cost
        upper_bound = ret
        for j in range(i, n):
            if stocks_sorted[j][1] <= remaining_budget:
                remaining_budget -= stocks_sorted[j][1]
                upper_bound += stocks_sorted[j][2]
            else:
                upper_bound += (remaining_budget / stocks_sorted[j][1]) * stocks_sorted[j][2]
                break
        if upper_bound < best_return:
            return
        if i == n:
            if ret > best_return:
                best_return = ret
                best_combo = combo[:]
            return
        # Include current stock
        combo.append(stocks_sorted[i])
        backtrack(i + 1, combo, cost + stocks_sorted[i][1], ret + stocks_sorted[i][2])
        combo.pop()
        # Exclude current stock
        backtrack(i + 1, combo, cost, ret)

    backtrack(0, [], 0, 0)
    total_cost = sum(s[1] for s in best_combo)
    return best_combo, total_cost, best_return

# =====================
# STREAMLIT INTERFACE
# =====================

st.set_page_config(page_title="Hybrid Stock Portfolio Optimizer", layout="wide")
st.title("📈 Hybrid Stock Portfolio Optimizer (Greedy + DP + Backtracking)")

st.markdown("""
Upload a CSV file with *Stock, **Price, and **ExpectedReturn* columns.  
The system will automatically optimize your portfolio using *Greedy, **Dynamic Programming, and **Backtracking*.
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
            # Step 1: Greedy
            greedy_sel, greedy_cost, greedy_ret = greedy_fractional(stocks, budget)
            st.subheader("⚡ Greedy (Fractional) Solution")
            st.table({
                "Stock": [s[0] for s in greedy_sel],
                "Invested ($)": [round(s[1], 2) for s in greedy_sel],
                "Expected Return": [round(s[2], 2) for s in greedy_sel]
            })
            st.info(f"Total Cost: ${greedy_cost:.2f} | Total Expected Return: {greedy_ret:.2f}")

            # Step 2: DP
            dp_sel, dp_cost, dp_ret = dp_knapsack(stocks, budget)
            st.subheader("🧠 Dynamic Programming (0/1) Solution")
            st.table({
                "Stock": [s[0] for s in dp_sel],
                "Invested ($)": [round(s[1], 2) for s in dp_sel],
                "Expected Return": [round(s[2], 2) for s in dp_sel]
            })
            st.info(f"Total Cost: ${dp_cost:.2f} | Total Expected Return: {dp_ret:.2f}")

            # Step 3: Backtracking
            bt_sel, bt_cost, bt_ret = backtracking_selection(stocks, budget, greedy_ret, dp_ret)
            st.subheader("🔍 Backtracking (Hybrid Optimal) Solution")
            st.table({
                "Stock": [s[0] for s in bt_sel],
                "Invested ($)": [round(s[1], 2) for s in bt_sel],
                "Expected Return": [round(s[2], 2) for s in bt_sel]
            })
            st.success(f"✅ Optimal Total Cost: ${bt_cost:.2f} | Optimal Total Return: {bt_ret:.2f}")

            # Comparison
            st.markdown("---")
            st.subheader("📊 Comparison Summary")
            comparison = {
                "Algorithm": ["Greedy", "Dynamic Programming", "Backtracking (Hybrid Optimal)"],
                "Total Cost ($)": [round(greedy_cost, 2), round(dp_cost, 2), round(bt_cost, 2)],
                "Total Return": [round(greedy_ret, 2), round(dp_ret, 2), round(bt_ret, 2)]
            }
            st.table(comparison)
else:
    st.info("📥 Please upload a CSV file to begin.")
