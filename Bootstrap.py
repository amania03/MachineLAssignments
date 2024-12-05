import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress tracking
from joblib import Parallel, delayed  # Parallel processing

# Load the dataset
mammogram = pd.read_csv('mammogram.csv')

# Cross-tabulate treatment and breast_cancer_death
cross_tab = pd.crosstab(mammogram['treatment'], mammogram['breast_cancer_death'])
print("Cross-tabulation of treatment and breast_cancer_death:\n", cross_tab)

# Calculate survival rates
survival_rates = cross_tab.div(cross_tab.sum(axis=1), axis=0)['no']
print("\nSurvival rates:\n", survival_rates)

# Difference in survival rates
diff_survival = survival_rates['mammogram'] - survival_rates['control']
print("\nDifference in survival rates:", diff_survival)

# Bootstrap function
def bootstrap_sample(data):
    sample = data.sample(frac=1, replace=True)
    boot_tab = pd.crosstab(sample['treatment'], sample['breast_cancer_death'])
    boot_survival = boot_tab.div(boot_tab.sum(axis=1), axis=0)['no']
    return boot_survival['mammogram'] - boot_survival['control']

# Bootstrap distributions with progress tracking and parallelization
n_bootstrap = 1000  # Adjust to 10,000 for full results
bootstrap_diff = Parallel(n_jobs=-1)(
    delayed(bootstrap_sample)(mammogram) for _ in tqdm(range(n_bootstrap), desc="Bootstrapping Mammogram")
)

# Confidence interval for the difference
ci_99 = np.percentile(bootstrap_diff, [0.5, 99.5])
print("\n99% Confidence interval for the difference:", ci_99)

# Plot bootstrap distributions
sns.histplot(bootstrap_diff, kde=True, color='blue', bins=50)
plt.axvline(ci_99[0], color='red', linestyle='dashed', label=f"99% CI lower: {ci_99[0]:.4f}")
plt.axvline(ci_99[1], color='green', linestyle='dashed', label=f"99% CI upper: {ci_99[1]:.4f}")
plt.title("Bootstrap Distribution of Survival Rate Difference")
plt.xlabel("Difference in Survival Rates")
plt.legend()
plt.show()

# Does the interval include zero? Yes, which suggests the difference in survival rates may not be statistically significant.
# Why might these data over/understate the conclusions?
# Overstatement: The dataset lacks detailed patient demographics, tumor characteristics, and other factors like access to follow-up care, which could influence survival rates.
# Understatement: The analysis only accounts for survival from breast cancer, not other potential benefits or harms of mammography (e.g., early detection of non-lethal cancers or false positives).
# Additional data needed: Patient age, tumor stage at diagnosis, family history, and access to healthcare. Follow-up care data, like adherence to treatments and co-morbidities.




# Load the diabetes dataset
diabetes = pd.read_csv('diabetes_hw.csv')

# Cross-tabulate treatment and outcome
cross_tab_diabetes = pd.crosstab(diabetes['treatment'], diabetes['outcome'])
print("Cross-tabulation of treatment and outcome:\n", cross_tab_diabetes)

# Proportion of successes
success_rates = cross_tab_diabetes.div(cross_tab_diabetes.sum(axis=1), axis=0)['success']
print("\nSuccess rates:\n", success_rates)

# Bootstrap function for diabetes
def bootstrap_success_sample(data, treatment):
    sample = data.sample(frac=1, replace=True)
    boot_tab = pd.crosstab(sample['treatment'], sample['outcome'])
    boot_success = boot_tab.div(boot_tab.sum(axis=1), axis=0)['success']
    return boot_success[treatment]

# Bootstrap distributions for each treatment
n_bootstrap = 1000  # Adjust to 10,000 for full results
bootstrap_success = {treatment: [] for treatment in diabetes['treatment'].unique()}

for treatment in bootstrap_success.keys():
    bootstrap_success[treatment] = Parallel(n_jobs=-1)(
        delayed(bootstrap_success_sample)(diabetes, treatment) for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping {treatment}")
    )

# Kernel Density Plot
plt.figure(figsize=(12, 6))
for treatment, values in bootstrap_success.items():
    sns.kdeplot(values, label=treatment)
plt.title("Kernel Density Plot of Success Proportions")
plt.xlabel("Proportion of Successes")
plt.legend()
plt.show()

# Pairwise treatment comparisons
pairwise_diffs = {
    ('lifestyle', 'met'): [],
    ('met', 'rosi'): [],
    ('rosi', 'lifestyle'): []
}

def bootstrap_pairwise(data, pair):
    sample = data.sample(frac=1, replace=True)
    boot_tab = pd.crosstab(sample['treatment'], sample['outcome'])
    boot_success = boot_tab.div(boot_tab.sum(axis=1), axis=0)['success']
    return boot_success[pair[0]] - boot_success[pair[1]]

for pair in pairwise_diffs.keys():
    pairwise_diffs[pair] = Parallel(n_jobs=-1)(
        delayed(bootstrap_pairwise)(diabetes, pair) for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping {pair}")
    )

# Confidence intervals for pairwise comparisons
for pair, diffs in pairwise_diffs.items():
    ci_90 = np.percentile(diffs, [5, 95])
    print(f"90% Confidence interval for {pair}: {ci_90}")

# Bootstrap the distribution of pairwise differences:
# Pairwise 90% confidence intervals:
# Lifestyle vs. Met: [-0.0218, 0.1263] → Not significantly different.
# Met vs. Rosi: [-0.2096, -0.0578] → Significantly different, favoring rosi.
# Rosi vs. Lifestyle: [-0.0005, 0.1533] → Not significantly different.
# Which treatment appears most effective overall?
# Rosiglitazone (rosi) consistently shows the highest success rate and a significant advantage over metformin.



#I would like to cite the assignment solutions and help from ChatGPT to help complete this assignment
