#StatTOPSIS
#Author: Mandeep Kumar
#Email: themandeepkumar@gmail.com

import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare
import seaborn as sns

# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Define the TOPSIS function with epsilon to avoid division by zero
def topsis(raw_data, weights, benefit_categories, epsilon=1e-10):
    m, n = raw_data.shape
    # Normalize the raw data
    divisors = np.sqrt(np.sum(raw_data ** 2, axis=0))
    normalized_data = raw_data / divisors

    # Apply weights
    weighted_data = normalized_data * weights

    # Determine Ideal and Negative Ideal Solutions
    ideal_solution = np.zeros(n)
    negative_ideal_solution = np.zeros(n)
    for j in range(n):
        if j in benefit_categories:
            ideal_solution[j] = np.max(weighted_data[:, j])
            negative_ideal_solution[j] = np.min(weighted_data[:, j])
        else:
            ideal_solution[j] = np.min(weighted_data[:, j])
            negative_ideal_solution[j] = np.max(weighted_data[:, j])

    # Calculate distances
    dist_to_ideal = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_data - negative_ideal_solution) ** 2, axis=1))

    # Calculate TOPSIS scores with epsilon to prevent division by zero
    scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal + epsilon)
    return scores

# Identification of Criteria and Weights
categories = np.array(["Maximum Memory (GB)", "Supported Languages", "Free Tier (million invocations)", "On-Demand USD Cost (per million invocations)"])
alternatives = np.array(["AWS Lambda", "Microsoft Azure Functions", "Google Cloud Functions", "Oracle Cloud Functions"])
raw_data = np.array([
    [10, 8, 1, 0.20],
    [15, 7, 1, 0.20],
    [32, 7, 2, 0.40],
    [2, 6, 2, 0.20],

])

initial_weights = np.array([0.25, 0.25, 0.25, 0.25])
benefit_categories = set([0, 1, 2])

# Display raw data and weights
raw_data_df = pd.DataFrame(data=raw_data, index=alternatives, columns=categories)
weights_df = pd.DataFrame(data=initial_weights, index=categories, columns=["Weights"])

print("Raw Data:")
display(raw_data_df)
print("Initial Weights:")
display(weights_df)

# Normalize the raw data
m, n = raw_data.shape
divisors = np.empty(n)
for j in range(n):
    column = raw_data[:, j]
    divisors[j] = np.sqrt(column @ column)
normalized_data = raw_data / divisors

# Normalize the weights to ensure that they sum up to 1
weights = initial_weights / np.sum(initial_weights)

normalized_data_df = pd.DataFrame(data=normalized_data, index=alternatives, columns=categories)

print("Normalized Data:")
display(normalized_data_df)

# Weighted normalized decision matrix
weighted_data = normalized_data * weights

weighted_data_df = pd.DataFrame(data=weighted_data, index=alternatives, columns=categories)

print("Weighted Normalized Data:")
display(weighted_data_df)

# Determine the Ideal and Negative Ideal Solutions
a_pos = np.zeros(n)
a_neg = np.zeros(n)
for j in range(n):
    column = weighted_data[:, j]
    max_val = np.max(column)
    min_val = np.min(column)

    if j in benefit_categories:
        a_pos[j] = max_val
        a_neg[j] = min_val
    else:
        a_pos[j] = min_val
        a_neg[j] = max_val

ideal_df = pd.DataFrame(data=[a_pos, a_neg], index=["A+", "A-"], columns=categories)
print("Ideal and Negative Ideal Solutions:")
display(ideal_df)

# Calculate the similarity scores
sp = np.zeros(m)
sn = np.zeros(m)
cs = np.zeros(m)

for i in range(m):
    diff_pos = weighted_data[i] - a_pos
    diff_neg = weighted_data[i] - a_neg
    sp[i] = np.sqrt(diff_pos @ diff_pos)
    sn[i] = np.sqrt(diff_neg @ diff_neg)
    cs[i] = sn[i] / (sp[i] + sn[i])

similarity_scores_df = pd.DataFrame(data=zip(sp, sn, cs), index=alternatives, columns=["S+", "S-", "Ci"])
print("Similarity Scores:")
display(similarity_scores_df)

# Ranking of alternatives
initial_ranks = rankdata(-cs)
ranking_df = pd.DataFrame(data=zip(cs, initial_ranks), index=alternatives, columns=["TOPSIS Score", "Initial Rank"]).sort_values(by="Initial Rank")
print("Initial Ranking of Alternatives (Descending Order):")
display(ranking_df)

# Sensitivity Analysis: Varying weights for each criterion
def sensitivity_analysis(raw_data, initial_weights, benefit_categories, alternatives):
    sensitivities = {}
    # Obtain initial ranking with current weights
    base_scores = topsis(raw_data, initial_weights, benefit_categories)
    base_ranking = rankdata(-base_scores)

    for i in range(len(initial_weights)):
        altered_weights = initial_weights.copy()
        for delta in np.linspace(-0.1, 0.1, 5):  # vary weights by ±10%
            if 0 <= initial_weights[i] + delta <= 1:
                altered_weights[i] = initial_weights[i] + delta
                # Ensure the weights sum to 1
                altered_weights /= np.sum(altered_weights)
                scores = topsis(raw_data, altered_weights, benefit_categories)
                ranking = rankdata(-scores)
                # Store the result using base_ranking as reference
                sensitivity_key = (i, delta)
                sensitivities[sensitivity_key] = pd.Series(ranking, index=alternatives)

    # Convert sensitivity results to DataFrame and align columns with initial ranking
    sensitivity_df = pd.DataFrame(sensitivities).T
    sensitivity_df.columns = alternatives  # Ensure correct column names for alternatives
    sensitivity_df.index.names = ['Criterion', 'Delta']
    sensitivity_df = sensitivity_df[ranking_df.sort_values("Initial Rank").index]

    return sensitivity_df

# Perform sensitivity analysis
sensitivity_df = sensitivity_analysis(raw_data, initial_weights, benefit_categories, alternatives)

print("Sensitivity Analysis:")
display(sensitivity_df)

# Bootstrapping Analysis: Generating bootstrap samples and calculating TOPSIS scores
def bootstrap_analysis(raw_data, initial_weights, benefit_categories, num_samples=1000):
    m, n = raw_data.shape
    bootstrap_scores = np.zeros((num_samples, m))

    for i in range(num_samples):
        bootstrap_sample_indices = np.random.choice(m, m, replace=True)
        bootstrap_sample = raw_data[bootstrap_sample_indices]
        bootstrap_scores[i] = topsis(bootstrap_sample, initial_weights, benefit_categories)

    return bootstrap_scores

bootstrap_scores = bootstrap_analysis(raw_data, initial_weights, benefit_categories)

# Analyzing the bootstrap results
bootstrap_ranks = np.array([rankdata(-scores) for scores in bootstrap_scores])
bootstrap_mean_ranks = np.mean(bootstrap_ranks, axis=0)
bootstrap_rank_intervals = np.percentile(bootstrap_ranks, [2.5, 97.5], axis=0)

# Display bootstrap analysis results
bootstrap_df = pd.DataFrame({
    "TOPSIS Score": topsis(raw_data, initial_weights, benefit_categories),
    "Initial Rank": initial_ranks,
    "Mean Rank": bootstrap_mean_ranks,
    "2.5% Rank": bootstrap_rank_intervals[0],
    "97.5% Rank": bootstrap_rank_intervals[1]
}, index=alternatives).sort_values(by="Initial Rank")

print("Bootstrap Analysis Results (Descending Order):")
display(bootstrap_df)

# Non-parametric Tests: Friedman Test
def friedman_test(bootstrap_ranks):
    # Perform the Friedman test
    stat, p = friedmanchisquare(*bootstrap_ranks.T)
    return stat, p

# Perform the Friedman test
stat, p = friedman_test(bootstrap_ranks)
print(f"Friedman Test Statistic: {stat}, p-value: {p}")

# Adding Friedman Test p-value to summary table
bootstrap_df["Friedman Test p-value"] = p
print("Final Summary Table with Friedman Test p-value (Descending Order):")
display(bootstrap_df)
