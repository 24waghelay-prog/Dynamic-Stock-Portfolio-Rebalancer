import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================
# ALGORITHM FUNCTIONS
# =====================

# Greedy Fractional Algorithm (CORRECTED - True Fractional Knapsack)
def greedy_fractional(stocks, budget):
    # Sort by return-to-price ratio (efficiency)
    stocks_sorted = sorted(stocks, key=lambda x: x[2]/x[1], reverse=True)
    total_cost = 0
    result = []
    
    for name, price, ret in stocks_sorted:
        if total_cost + price <= budget:
            # Can afford the whole stock
            result.append((name, price, ret))
            total_cost += price
        else:
            # Take fractional amount of this stock
            remaining_budget = budget - total_cost
            fraction = remaining_budget / price
            result.append((name, remaining_budget, ret * fraction))
            total_cost += remaining_budget
            break  # Budget exhausted
    
    total_return = sum(r[2] for r in result)
    return result, total_cost, total_return


# Dynamic Programming (0/1 Knapsack) - NO fractions allowed
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


# Backtracking (Exhaustive Search) - NO fractions allowed
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
st.title("🎯 Dynamic Stock Portfolio Optimizer")

st.markdown("""
Upload a CSV file with **Stock**, **Price**, and **ExpectedReturn** columns.  
Compare three optimization approaches:
- **🟢 Greedy Fractional:** Allows buying fractions of stocks (highest efficiency first)
- **🔵 Dynamic Programming:** 0/1 Knapsack - buy whole stocks only (optimal for 0/1)
- **🟣 Backtracking:** Exhaustive search - buy whole stocks only (guaranteed optimal)
""")

uploaded_file = st.file_uploader("Upload your stock dataset (CSV file)", type=["csv"])
budget = st.number_input("Enter your total investment budget ($):", value=5000, min_value=100)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Stock", "Price", "ExpectedReturn"}

    if not required_columns.issubset(df.columns):
        st.error("❌ CSV must contain 'Stock', 'Price', and 'ExpectedReturn' columns.")
    else:
        stocks = [tuple(x) for x in df[["Stock", "Price", "ExpectedReturn"]].values]
        
        # Add efficiency ratio column for display
        df['Efficiency (Return/Price)'] = (df['ExpectedReturn'] / df['Price']).round(4)
        
        st.write("### 📊 Loaded Stocks:")
        st.dataframe(df)

        if st.button("🚀 Run Hybrid Optimization"):
            col1, col2, col3 = st.columns(3)
            
            # Greedy Algorithm
            with col1:
                greedy_sel, greedy_cost, greedy_ret = greedy_fractional(stocks, budget)
                st.subheader("🟢 Greedy (Fractional)")
                greedy_df = pd.DataFrame(greedy_sel, columns=["Stock", "Invested ($)", "Expected Return"])
                greedy_df['Type'] = ['Full' if row['Invested ($)'] == df[df['Stock'] == row['Stock']]['Price'].values[0] 
                                     else 'Fractional' for _, row in greedy_df.iterrows()]
                st.dataframe(greedy_df)
                st.info(f"**Cost:** ${greedy_cost:.2f} | **Return:** {greedy_ret:.2f}")

            # Dynamic Programming
            with col2:
                dp_sel, dp_cost, dp_ret = dp_knapsack(stocks, budget)
                st.subheader("🔵 Dynamic Programming")
                if dp_sel:
                    st.dataframe(pd.DataFrame(dp_sel, columns=["Stock", "Price ($)", "Expected Return"]))
                else:
                    st.write("No stocks selected")
                st.info(f"**Cost:** ${dp_cost:.2f} | **Return:** {dp_ret:.2f}")

            # Backtracking (Exhaustive Search)
            with col3:
                st.subheader("🟣 Backtracking")
                if len(stocks) > 25:
                    st.warning(f"⚠️ Dataset too large ({len(stocks)} stocks)! Backtracking skipped.\n\nBacktracking tries all 2^{len(stocks)} combinations. Use ≤25 stocks for backtracking.")
                    bt_sel, bt_cost, bt_ret = [], 0, 0
                else:
                    with st.spinner("Running exhaustive search..."):
                        bt_sel, bt_cost, bt_ret = backtracking_selection(stocks, budget)
                    if bt_sel:
                        st.dataframe(pd.DataFrame(bt_sel, columns=["Stock", "Price ($)", "Expected Return"]))
                    else:
                        st.write("No stocks selected")
                    st.success(f"**Cost:** ${bt_cost:.2f} | **Return:** {bt_ret:.2f}")

            # Comparison Summary
            st.markdown("---")
            st.subheader("📈 Comparison Summary")
            
            if len(stocks) <= 25:
                comparison_df = pd.DataFrame({
                    "Algorithm": ["Greedy Fractional", "Dynamic Programming (0/1)", "Backtracking (0/1)"],
                    "Total Cost ($)": [round(greedy_cost, 2), round(dp_cost, 2), round(bt_cost, 2)],
                    "Total Return": [round(greedy_ret, 2), round(dp_ret, 2), round(bt_ret, 2)],
                    "Allows Fractions": ["✅ Yes", "❌ No", "❌ No"]
                })
            else:
                comparison_df = pd.DataFrame({
                    "Algorithm": ["Greedy Fractional", "Dynamic Programming (0/1)"],
                    "Total Cost ($)": [round(greedy_cost, 2), round(dp_cost, 2)],
                    "Total Return": [round(greedy_ret, 2), round(dp_ret, 2)],
                    "Allows Fractions": ["✅ Yes", "❌ No"]
                })
            st.dataframe(comparison_df, use_container_width=True)

            # Chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Return comparison
            colors = ['#28a745', '#007bff', '#6f42c1']
            if len(stocks) <= 25:
                algorithms = ["Greedy\nFractional", "Dynamic\nProgramming", "Backtracking"]
                returns = [greedy_ret, dp_ret, bt_ret]
                costs = [greedy_cost, dp_cost, bt_cost]
            else:
                algorithms = ["Greedy\nFractional", "Dynamic\nProgramming"]
                returns = [greedy_ret, dp_ret]
                costs = [greedy_cost, dp_cost]
                colors = colors[:2]
            
            ax1.bar(algorithms, returns, color=colors)
            ax1.set_ylabel("Total Expected Return")
            ax1.set_title("Expected Returns Comparison")
            ax1.grid(axis='y', alpha=0.3)
            
            # Cost comparison
            ax2.bar(algorithms, costs, color=colors)
            ax2.axhline(y=budget, color='r', linestyle='--', label=f'Budget: ${budget}')
            ax2.set_ylabel("Total Cost ($)")
            ax2.set_title("Cost Utilization Comparison")
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Key Insights
            st.markdown("---")
            st.subheader("🔍 Key Insights")
            if len(stocks) <= 25:
                st.markdown(f"""
                - **Greedy Fractional** can buy partial stocks, so it typically achieves the highest return and uses the full budget
                - **Dynamic Programming & Backtracking** only buy whole stocks (0/1), so they might leave unused budget
                - For 0/1 problems, **DP and Backtracking should give identical results** (both optimal)
                - Greedy is **fastest** O(n log n)
                - DP is **fast** O(n × budget)
                - Backtracking is **slowest** O(2^n) - tries all {2**len(stocks):,} combinations!
                """)
            else:
                st.markdown(f"""
                - **Greedy Fractional** can buy partial stocks, so it typically achieves the highest return and uses the full budget
                - **Dynamic Programming** only buys whole stocks (0/1), so it might leave unused budget
                - **Backtracking skipped** - would need to try 2^{len(stocks)} = {2**len(stocks):,} combinations! (use ≤25 stocks)
                - Greedy is **fastest** O(n log n)
                - DP is **fast** O(n × budget) - optimal for large datasets with 0/1 constraint
                """)
else:
    st.info("📁 Please upload a CSV file to begin.")
