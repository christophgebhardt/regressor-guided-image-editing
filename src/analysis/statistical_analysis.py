from scipy.stats import shapiro, kruskal
import scikit_posthocs as sp
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def one_way_stats_test(df, iv, dv):
    statistic1, p_value1 = shapiro(df[dv].values)
    if p_value1 < 0.05:
        # print(f"Data does not appear to be normally distributed.")
        kruskal_wallis_test(df, iv, dv)
    else:
        # print(f"Data does appear to be normally distributed.")
        one_way_anova(df, iv, dv)


def one_way_anova(df, iv, dv):
    # Fit the ANOVA model
    model = ols(f'{dv} ~ C({iv})', data=df).fit()

    # Perform ANOVA analysis
    anova_table = anova_lm(model, typ=2)

    # Display the ANOVA table
    print("\n" + dv)
    print("ANOVA Table:")
    print(anova_table)
    print_mean_sd(df, iv, dv)

    # Access the p-value
    p_value = anova_table['PR(>F)'][f"C({iv})"]
    if p_value < 0.05:
        post_hoc_ttest(df, iv, dv)


def post_hoc_ttest(df, iv, dv):
    # res = sp.posthoc_ttest(df, val_col=dv, group_col=iv, p_adjust='holm')
    res = sp.posthoc_ttest(df, val_col=dv, group_col=iv, p_adjust='bonferroni')
    print(res)


def kruskal_wallis_test(df, iv, dv):
    # Perform the Kruskal-Wallis test
    statistic, p_value = kruskal(*[df[df[iv] == level][dv] for level in df[iv].unique()])

    # Print the test result
    print("\n" + dv)
    print(f"Kruskal-Wallis Test Statistic: {statistic}")
    print(f"P-value: {p_value}")
    print_mean_sd(df, iv, dv)

    # Determine if the result is statistically significant
    alpha = 0.05  # Set your significance level
    if p_value < alpha:
        # Perform post-hoc pairwise comparisons (using Conover's post hoc)
        posthoc_result = sp.posthoc_conover(df, val_col=dv, group_col=iv, p_adjust='holm')

        print("Post-Hoc Test Results:")
        print(posthoc_result)


def print_mean_sd(df, iv, dv):
    unique_ivs = df[iv].unique()
    print()
    for iv_l in unique_ivs:
        rows = df[df[iv] == iv_l]
        print(f"{iv_l} M:{rows[dv].mean():.2f} SD:{rows[dv].std():.2f}")
    print()

