import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import norm
from prettytable import PrettyTable

class AnomalyDetection:

  def __init__(self, model_df, target):
    self.data = pd.DataFrame({'DateTime': model_df['DateTime'], 'Actual': model_df[target], 'Predicted': model_df['pred']})
    self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
    self.target = target
    self.errors = model_df['error']
    self.actual = self.data['Actual']
    self.predicted = self.data['Predicted']
    self.anomalies = pd.DataFrame()
    self.statistical_detection()
    self.gmm()

  def statistical_detection(self):
    num_stds = 2

    mean = self.errors.mean()
    std = self.errors.std()
    upper_threshold = mean + num_stds * std
    lower_threshold = mean - num_stds * std

    self.anomalies = self.data[(self.errors > upper_threshold) | (self.errors < lower_threshold)]

    #plt.figure(figsize=(10, 6))
    #plt.scatter(self. predicted, self.actual, label='Normal data', c='blue', alpha=0.7)
    #plt.scatter(self.anomalies['Predicted'], self.anomalies['Actual'], label='Anomalies', c='red', edgecolors='k')
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')
    #plt.title(f'Statistical anomaly detection with num_stds={num_stds}')
    #plt.legend()
    #plt.show()

  def gmm(self, ax=None):
    errors = self.errors.values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=1, random_state=42).fit(errors)

    log_likelihood = gmm.score_samples(errors)      # how well the model explains observed data
    threshold = np.percentile(log_likelihood, 5)
    gmm_anomalies = self.data[log_likelihood < threshold]

    common_index = gmm_anomalies.index.intersection(self.anomalies.index)
    self.anomalies = gmm_anomalies.loc[common_index]

    p_vals = np.exp(log_likelihood[self.anomalies.index])
    delta_x = (3.5*errors.std())/(len(errors)**(1/3))
    approx_p = p_vals * delta_x
    norm_scores = 1 - (approx_p - approx_p.min()) / (approx_p.max() - approx_p.min())
    self.anomalies['Anomaly Score'] = norm_scores

  # plots scatter plot of anomalies and normal data points
  def final_plot(self, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    ax.scatter(self.predicted, self.actual, label='Normal data', c='blue', alpha=0.7)
    sc = ax.scatter(self.anomalies['Predicted'], self.anomalies['Actual'], c=self.anomalies['Anomaly Score'], cmap='viridis', edgecolors='k', s=50, label='Anomalies')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Anomaly Likelihood')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Final Anomalies')
    ax.legend()

  # prints number of anomalies detected
  def num_anomalies(self):
    print(f'\n{len(self.anomalies)} final anomalies detected out of {len(self.data)} data points using Gaussian mixture modelling and statistical calculations.\n')

  # prints table of best ten anomalies (highest anomaly scores)
  def best_ten_anomalies(self):
    table = PrettyTable(["DateTime", "Toronto", "Predicted", "Anomaly Score"])
    best_ten = self.anomalies.sort_values(by=['Anomaly Score'], ascending=True)[0:10]
    table.title = "Best ten anomalies"
    for index, row in best_ten.iterrows():
      table.add_row([row['DateTime'], row['Actual'], row['Predicted'], row['Anomaly Score']])
    print(table)

  # prints table of worst ten anomalies (lowest anomaly scores)
  def worst_ten_anomalies(self):
    table = PrettyTable(["DateTime", "Toronto", "Predicted", "Anomaly Score"])
    worst_ten = self.anomalies.sort_values(by=['Anomaly Score'], ascending=False)[0:10]
    table.title = "Worst ten anomalies"
    for index, row in worst_ten.iterrows():
      table.add_row([row['DateTime'], row['Actual'], row['Predicted'], row['Anomaly Score']])
    print(table)

  # plots longest streak of consecutive anomalies
  def longest_anomalous_streak(self, ax=None):
    self.anomalies['group'] = (self.anomalies.index.to_series().diff() != 1).cumsum()
    longest_group = self.anomalies['group'].value_counts().idxmax()
    longest_scores = self.anomalies[self.anomalies['group'] == longest_group]
    longest_scores_idx = longest_scores.index
    consecutive_anomalies = self.anomalies.loc[longest_scores_idx]
    hours = consecutive_anomalies.index - consecutive_anomalies.index[0]
    consecutive_anomalies['Hour'] = hours

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    ax.plot(consecutive_anomalies['DateTime'], consecutive_anomalies['Actual'], label='Actual')
    ax.plot(consecutive_anomalies['DateTime'], consecutive_anomalies['Predicted'], label='Predicted')

    xticks = np.arange(consecutive_anomalies.index[0], consecutive_anomalies.index[len(consecutive_anomalies)-1], 2)
    ax.set_xticks(consecutive_anomalies['DateTime'][xticks])
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Power')
    ax.set_title(f"Most consecutive anomalies on {consecutive_anomalies['DateTime'][consecutive_anomalies.index[0]]}")
    ax.legend()

  # plots longest streak of consecutive nonanomalous points
  def longest_nonanomalous_streak(self, ax=None):
    differences = self.anomalies.index.to_series().diff()
    max_diff_idx = int(differences.idxmax())
    prev_idx = int(max_diff_idx - differences.loc[max_diff_idx])
    total_days = (max_diff_idx - prev_idx) / 24
    tick_interval = 48 if total_days > 7 else 24

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    ax.plot(self.data['DateTime'][prev_idx:max_diff_idx+1], self.data['Actual'][prev_idx:max_diff_idx+1], label='Actual')
    ax.plot(self.data['DateTime'][prev_idx:max_diff_idx+1], self.data['Predicted'][prev_idx:max_diff_idx+1], label='Predicted')

    ax.set_xlabel('DateTime')
    ax.set_ylabel('Power')
    ax.set_title(f"Longest period without an anomaly: {self.data.loc[prev_idx]['DateTime']} to {self.data.loc[max_diff_idx]['DateTime']}")

    xticks = np.arange(prev_idx, max_diff_idx + 1, tick_interval)
    ax.set_xticks(self.data['DateTime'][xticks])
    ax.tick_params(axis='x', rotation=45)

    ax.legend()

  # plots anomaly frequency per month
  def anomalies_per_month(self, ax=None):
    anomaly_counts = self.anomalies['DateTime'].dt.month.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.month.value_counts().sort_index()

    normalized_counts = anomaly_counts / total_counts


    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    ax.bar(normalized_counts.index, normalized_counts.values)
    ax.set_xlabel('Month')
    ax.set_ylabel('Proportion of anomalies')
    ax.set_title('Anomaly Frequency by Month (normalized)')
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, normalized_counts.max() + 0.05)

  # plots anomaly frequency per day
  def anomalies_per_day(self):
    anomaly_counts = self.anomalies['DateTime'].dt.day.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.day.value_counts().sort_index()
    normalized_counts = anomaly_counts / total_counts
    plt.figure(figsize=(10, 6))
    plt.bar(normalized_counts.index, normalized_counts.values)
    plt.xlabel('Day')
    plt.ylabel('Proportion of anomalies')
    plt.title('Anomaly Frequency by Day (normalized)')
    plt.xticks(range(1, 32))  # Ensure days are labeled 1 to 31
    plt.ylim(0, normalized_counts.max() + 0.05)  # Since it's a proportion
    plt.show()

  # plots anomaly frequency per hour
  def anomalies_per_hour(self):
    anomaly_counts = self.anomalies['DateTime'].dt.hour.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.hour.value_counts().sort_index()
    normalized_counts = anomaly_counts / total_counts
    plt.figure(figsize=(10, 6))
    plt.bar(normalized_counts.index, normalized_counts.values)
    plt.xlabel('Hour')
    plt.ylabel('Proportion of anomalies')
    plt.title('Anomaly Frequency by Hour (normalized)')
    plt.xticks(range(24))
    plt.ylim(0, normalized_counts.max() + 0.05)
    plt.show()

  # plots four main plots
  def summary_plots(self):
    import io, base64
    plots = {}

    # Scatter Plot
    plt.figure(figsize=(8, 6))
    self.final_plot()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['scatter_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Anomalies per Month
    plt.figure(figsize=(8, 6))
    self.anomalies_per_month()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['month_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Longest Anomalous Streak
    plt.figure(figsize=(8, 6))
    self.longest_anomalous_streak()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['longest_anomalous_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Longest Non-anomalous Streak
    plt.figure(figsize=(8, 6))
    self.longest_nonanomalous_streak()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['longest_clean_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return plots
