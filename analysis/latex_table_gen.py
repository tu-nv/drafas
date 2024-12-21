
# New data input
data_updated = {
    "Use Case": ["Image Classification", "Chatbot", "Text to Speech"],
    "DRL SLO Violation Rate": [0.0018, 0.0024, 0.0064],
    "Rule SLO Violation Rate": [0.0017, 0.0034, 0.0029],
    "DRL Num Instances": [1.5115, 1.8771, 1.7219],
    "Rule Num Instances": [1.6538, 2.0104, 1.9764],
    "DRL Processing Time": [0.0707, 0.2956, 0.2735],
    "Rule Processing Time": [0.0707, 0.2955, 0.2672],
    "DRL Reward": [-0.02052, -0.02559, -0.02727],
    "Rule Reward": [-0.02221, -0.02819, -0.02733],
}

df_updated = pd.DataFrame(data_updated)

# Recalculate improvements
improvements_updated = {
    "SLO Violation Rate": ((df_updated["Rule SLO Violation Rate"] - df_updated["DRL SLO Violation Rate"]) / df_updated["Rule SLO Violation Rate"]) * 100,
    "Num Instances": ((df_updated["Rule Num Instances"] - df_updated["DRL Num Instances"]) / df_updated["Rule Num Instances"]) * 100,
    "Processing Time": ((df_updated["Rule Processing Time"] - df_updated["DRL Processing Time"]) / df_updated["Rule Processing Time"]) * 100,
    "Reward": ((df_updated["DRL Reward"] - df_updated["Rule Reward"]) / df_updated["Rule Reward"]) * 100,
}

# Generate the updated LaTeX table with horizontal line separation between use cases
latex_table_updated = """
\\begin{table}[h!]
\\centering
\\renewcommand{\\arraystretch}{1.2} % Set line spacing to 1.2
\\begin{tabular}{|l|l|l|l|}
\\hline
\\textbf{Use Case} & \\textbf{Metric} & \\textbf{DRL} & \\textbf{Rule-based} \\\\ \\hline
"""

for i, row in df_updated.iterrows():
    for j, (metric, drl, rule, improvement) in enumerate(zip(
        ["SLO Violation Rate", "Num Instances", "Processing Time", "Reward"],
        [row["DRL SLO Violation Rate"], row["DRL Num Instances"], row["DRL Processing Time"], row["DRL Reward"]],
        [row["Rule SLO Violation Rate"], row["Rule Num Instances"], row["Rule Processing Time"], row["Rule Reward"]],
        [improvements_updated[key][i] for key in ["SLO Violation Rate", "Num Instances", "Processing Time", "Reward"]]
    )):
        better_drl = (drl < rule if metric in ["SLO Violation Rate", "Num Instances", "Processing Time"] else drl > rule)
        drl_fmt = f"\\textbf{{{drl:.4f} ({improvement:.2f}\\%)}}" if better_drl else f"{drl:.4f}"
        rule_fmt = f"\\textbf{{{rule:.4f} ({-improvement:.2f}\\%)}}" if not better_drl else f"{rule:.4f}"
        if j == 0:  # Add use case only for the first metric
            latex_table_updated += f"\\multirow{{4}}{{*}}{{{row['Use Case']}}} & {metric} & {drl_fmt} & {rule_fmt} \\\\ \\cline{{2-4}}\n"
        else:
            latex_table_updated += f"& {metric} & {drl_fmt} & {rule_fmt} \\\\ \\cline{{2-4}}\n"
    latex_table_updated += "\\hline\n"  # Add horizontal line between use cases

latex_table_updated += """
\\end{tabular}
\\caption{Updated Comparison of DRL and Rule-based Agent Performance}
\\label{tab:updated_comparison}
\\end{table}
"""

print(latex_table_updated)
