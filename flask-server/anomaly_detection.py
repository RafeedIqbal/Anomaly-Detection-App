import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import norm
from prettytable import PrettyTable

class AnomalyDetection:

  # Constructor modified: Removed 'target' argument
  def __init__(self, model_df):
    # Ensure model_df has the expected structure (at least 3 columns: DateTime, Target, Pred)
    if len(model_df.columns) < 3:
        raise ValueError("Input DataFrame for AnomalyDetection must have at least 3 columns (e.g., DateTime, Target, Predicted).")

    # Infer target column name from the second column of the input model_df
    self.target = model_df.columns[1]
    self.first_col_name = model_df.columns[0] # Usually 'DateTime'
    print(f"Inferred target column for Anomaly Detection: '{self.target}'") # Added for confirmation

    # Reconstruct internal data using inferred target name for 'Actual'
    self.data = pd.DataFrame({
        'DateTime': model_df[self.first_col_name],
        'Actual': model_df[self.target], # Use inferred target name here
        'Predicted': model_df['pred'] # Assumes 3rd column is 'pred'
    })
    self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
    # self.target already set above
    self.errors = model_df['error'] # Assumes 4th column is 'error'
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

    self.anomalies = self.data[(self.errors > upper_threshold) | (self.errors < lower_threshold)].copy() # Use .copy()

    # Optional plotting (commented out by default)
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

    # Avoid GMM error if too few errors or no variance
    if len(errors) < 2 or errors.std() == 0:
         print("Warning: Not enough data or zero variance in errors for GMM. Skipping GMM step.")
         # Assign anomaly score of 1 (least anomalous) if skipping
         self.anomalies['Anomaly Score'] = 1.0
         return

    gmm = GaussianMixture(n_components=1, random_state=42).fit(errors)

    log_likelihood = gmm.score_samples(errors)      # how well the model explains observed data
    threshold = np.percentile(log_likelihood, 5) # Detect 5% most unlikely points based on error distribution
    gmm_anomalies_indices = np.where(log_likelihood < threshold)[0] # Get indices based on original errors array
    gmm_anomalies = self.data.iloc[gmm_anomalies_indices] # Select from self.data using original indices

    # Filter the statistically detected anomalies further using GMM results
    common_index = gmm_anomalies.index.intersection(self.anomalies.index)
    self.anomalies = self.anomalies.loc[common_index].copy() # Ensure it's a copy

    # Calculate anomaly scores only for the final intersecting anomalies
    if not self.anomalies.empty:
        final_anomalies_indices = self.anomalies.index # Get indices of final anomalies
        # Ensure log_likelihood corresponds to the full dataset length before indexing
        if len(log_likelihood) == len(self.data):
             p_vals = np.exp(log_likelihood[final_anomalies_indices])
             # Use errors corresponding to the final anomalies for delta_x calculation if needed,
             # but using the overall error distribution's std might be more robust.
             delta_x = (3.5 * self.errors.std()) / (len(self.errors)**(1/3)) if len(self.errors) > 0 else 1.0

             # Avoid division by zero if all p_vals are the same
             if p_vals.max() - p_vals.min() > 1e-9 : # Add tolerance for floating point comparison
                 approx_p = p_vals * delta_x
                 norm_scores = 1 - (approx_p - approx_p.min()) / (approx_p.max() - approx_p.min())
             else:
                 norm_scores = np.ones(len(self.anomalies)) # Assign neutral score if all p_vals are identical

             self.anomalies['Anomaly Score'] = norm_scores
        else:
             print("Warning: Mismatch between log_likelihood length and data length. Skipping anomaly score calculation.")
             self.anomalies['Anomaly Score'] = np.nan # Or some default
    else:
        # If no anomalies remain after intersection, ensure the column exists but is empty
        self.anomalies['Anomaly Score'] = pd.Series(dtype=float)


  # plots scatter plot of anomalies and normal data points
  def final_plot(self, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes

    ax.scatter(self.predicted, self.actual, label='Normal data', c='blue', alpha=0.7, s=10) # Smaller normal points

    # Check if there are any anomalies to plot and if 'Anomaly Score' column exists
    if not self.anomalies.empty and 'Anomaly Score' in self.anomalies.columns:
        # Drop rows with NaN scores if any exist before plotting
        plot_anomalies = self.anomalies.dropna(subset=['Anomaly Score'])
        if not plot_anomalies.empty:
             sc = ax.scatter(
                 plot_anomalies['Predicted'],
                 plot_anomalies['Actual'],
                 c=plot_anomalies['Anomaly Score'],
                 cmap='viridis_r', # Reversed viridis: yellow=high score (more anomalous)
                 edgecolors='k',
                 s=50,
                 label='Anomalies',
                 vmin=0, # Ensure colorbar starts at 0
                 vmax=1  # Ensure colorbar ends at 1
            )
             # Only add colorbar if scatter plot was created
             cbar = plt.colorbar(sc, ax=ax)
             cbar.set_label('Anomaly Score (1 = High Likelihood)') # Clarify score meaning
        else:
             print("No anomalies with valid scores to plot.")
    else:
        print("No anomalies found or 'Anomaly Score' column missing.")


    ax.set_xlabel('Predicted')
    ax.set_ylabel(f'Actual ({self.target})') # Use inferred target name
    ax.set_title('Anomaly Detection Results')
    ax.legend()
    # If ax was passed, don't call plt.show(). Let the caller handle it.
    # if ax is None: plt.show() # Only show if created internally


  # prints number of anomalies detected
  def num_anomalies(self):
    num_final_anomalies = len(self.anomalies)
    print(f'\n{num_final_anomalies} final anomalies detected out of {len(self.data)} data points using statistical filtering and Gaussian Mixture Model likelihood.\n')
    return num_final_anomalies # Return the number

  # prints table of best ten anomalies (highest anomaly scores)
  def best_ten_anomalies(self):
    # Check if anomalies exist and have scores
    if self.anomalies.empty or 'Anomaly Score' not in self.anomalies.columns:
        print("No anomalies found or scores not calculated.")
        return "No anomalies found."

    # Use inferred target name in the header
    table = PrettyTable(["DateTime", self.target, "Predicted", "Anomaly Score"])
    # Sort by Anomaly Score descending (higher score = more anomalous)
    best_ten = self.anomalies.sort_values(by=['Anomaly Score'], ascending=False).head(10)
    table.title = f"Top 10 Anomalies (Highest Score for Target: {self.target})"
    table.float_format = ".4" # Format floats
    for index, row in best_ten.iterrows():
      # Access 'Actual' column which contains the target data
      table.add_row([row['DateTime'].strftime('%Y-%m-%d %H:%M'), row['Actual'], row['Predicted'], row['Anomaly Score']])
    print(table)
    return table.get_string() # Return table as string

  # prints table of worst ten anomalies (lowest anomaly scores, closer to normal)
  def worst_ten_anomalies(self):
     # Check if anomalies exist and have scores
    if self.anomalies.empty or 'Anomaly Score' not in self.anomalies.columns:
        print("No anomalies found or scores not calculated.")
        return "No anomalies found."

    # Use inferred target name in the header
    table = PrettyTable(["DateTime", self.target, "Predicted", "Anomaly Score"])
     # Sort by Anomaly Score ascending (lower score = less anomalous)
    worst_ten = self.anomalies.sort_values(by=['Anomaly Score'], ascending=True).head(10)
    table.title = f"Bottom 10 Anomalies (Lowest Score for Target: {self.target})"
    table.float_format = ".4" # Format floats
    for index, row in worst_ten.iterrows():
       # Access 'Actual' column which contains the target data
      table.add_row([row['DateTime'].strftime('%Y-%m-%d %H:%M'), row['Actual'], row['Predicted'], row['Anomaly Score']])
    print(table)
    return table.get_string() # Return table as string


  # plots longest streak of consecutive anomalies
  def longest_anomalous_streak(self, ax=None):
    if self.anomalies.empty:
        print("No anomalies found to plot streak.")
        # Optionally draw an empty plot if ax is provided
        if ax:
            ax.text(0.5, 0.5, 'No anomalies found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title("Longest Anomalous Streak")
        return

    # Calculate consecutive groups
    self.anomalies['group'] = (self.anomalies.index.to_series().diff() != 1).cumsum()
    # Find the group with the most occurrences
    longest_group_id = self.anomalies['group'].value_counts().idxmax()
    # Select the anomalies belonging to that group
    consecutive_anomalies = self.anomalies[self.anomalies['group'] == longest_group_id].copy() # Use .copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(consecutive_anomalies['DateTime'], consecutive_anomalies['Actual'], label=f'Actual ({self.target})') # Use target name
    ax.plot(consecutive_anomalies['DateTime'], consecutive_anomalies['Predicted'], label='Predicted')

    # Improve x-axis labeling for streaks
    ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # Limit number of ticks
    ax.tick_params(axis='x', rotation=30)

    ax.set_xlabel('DateTime')
    ax.set_ylabel('Value') # Generic ylabel
    start_date_str = consecutive_anomalies['DateTime'].iloc[0].strftime('%Y-%m-%d %H:%M')
    ax.set_title(f"Longest Consecutive Anomaly Streak (Starting: {start_date_str})")
    ax.legend()
    plt.tight_layout() # Adjust layout


  # plots longest streak of consecutive nonanomalous points
  def longest_nonanomalous_streak(self, ax=None):
    if self.anomalies.empty:
        # If no anomalies, the whole dataset is non-anomalous
        start_idx = self.data.index[0]
        end_idx = self.data.index[-1]
        print("No anomalies found; plotting entire dataset as non-anomalous period.")
    elif len(self.anomalies) == len(self.data):
        print("All data points are anomalies; cannot plot non-anomalous streak.")
        if ax:
             ax.text(0.5, 0.5, 'All points are anomalous', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title("Longest Non-Anomalous Streak")
        return
    else:
        # Calculate differences between anomaly indices
        differences = self.anomalies.index.to_series().diff().fillna(self.anomalies.index[0] + 1) # Handle first potential streak
        # Find the index *after* the largest gap (this marks the start of the next anomaly)
        max_diff_idx_loc = differences.idxmax() # Index label where the largest gap ENDS
        # The largest gap value itself tells us the length of the non-anomalous period
        max_diff_val = differences.loc[max_diff_idx_loc]

        # Find the end of the non-anomalous streak (index of the anomaly *before* the gap)
        # Find the location of max_diff_idx_loc in the anomalies index
        try:
            loc_in_index = self.anomalies.index.get_loc(max_diff_idx_loc)
            if loc_in_index == 0: # The first anomaly has the largest difference from theoretical start
                 start_idx = self.data.index[0]
                 end_idx = self.anomalies.index[0] - 1
            else:
                 start_idx = self.anomalies.index[loc_in_index - 1] + 1
                 end_idx = max_diff_idx_loc - 1
        except KeyError:
             # Should not happen if max_diff_idx_loc comes from the index
             print("Error finding index location. Cannot plot non-anomalous streak.")
             return

        # Also consider the period after the last anomaly
        last_anomaly_idx = self.anomalies.index[-1]
        streak_after_last = len(self.data) - 1 - last_anomaly_idx
        if streak_after_last > (end_idx - start_idx +1) : # Compare lengths
             start_idx = last_anomaly_idx + 1
             end_idx = self.data.index[-1]

    # Ensure start and end indices are valid within self.data
    start_idx = max(start_idx, self.data.index[0])
    end_idx = min(end_idx, self.data.index[-1])

    if start_idx > end_idx:
        print("Could not determine a valid non-anomalous period.")
        if ax:
             ax.text(0.5, 0.5, 'No non-anomalous period found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title("Longest Non-Anomalous Streak")
        return


    # Select the data for the longest non-anomalous streak
    non_anomalous_data = self.data.loc[start_idx:end_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(non_anomalous_data['DateTime'], non_anomalous_data['Actual'], label=f'Actual ({self.target})')
    ax.plot(non_anomalous_data['DateTime'], non_anomalous_data['Predicted'], label='Predicted')

    ax.set_xlabel('DateTime')
    ax.set_ylabel('Value')
    start_date_str = non_anomalous_data['DateTime'].iloc[0].strftime('%Y-%m-%d %H:%M')
    end_date_str = non_anomalous_data['DateTime'].iloc[-1].strftime('%Y-%m-%d %H:%M')
    ax.set_title(f"Longest Non-Anomalous Period: {start_date_str} to {end_date_str}")

    ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # Limit ticks
    ax.tick_params(axis='x', rotation=30)
    ax.legend()
    plt.tight_layout()


  # plots anomaly frequency per month
  def anomalies_per_month(self, ax=None):
    if self.anomalies.empty:
        print("No anomalies found to plot frequency by month.")
        if ax:
            ax.text(0.5, 0.5, 'No anomalies found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Anomaly Frequency by Month (Normalized)')
            ax.set_xticks(range(1, 13))
            ax.set_xlabel('Month')
            ax.set_ylabel('Proportion of anomalies')
        return

    anomaly_counts = self.anomalies['DateTime'].dt.month.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.month.value_counts().sort_index()

    # Ensure all months are present in both series for division
    all_months = range(1, 13)
    anomaly_counts = anomaly_counts.reindex(all_months, fill_value=0)
    total_counts = total_counts.reindex(all_months, fill_value=0)

    # Calculate normalized counts, handle division by zero if a month has no data points
    normalized_counts = anomaly_counts.divide(total_counts).fillna(0)


    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(normalized_counts.index, normalized_counts.values)
    ax.set_xlabel('Month')
    ax.set_ylabel('Proportion of Anomalies')
    ax.set_title('Anomaly Frequency by Month (Normalized)')
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, max(normalized_counts.max() * 1.1, 0.05)) # Adjust ylim, ensure minimum height
    # Add month names for clarity
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


  # plots anomaly frequency per day (of the month)
  def anomalies_per_day(self, ax=None): # Added ax parameter
    if self.anomalies.empty:
        print("No anomalies found to plot frequency by day.")
        if ax:
            ax.text(0.5, 0.5, 'No anomalies found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Anomaly Frequency by Day (Normalized)')
            ax.set_xticks(range(1, 32))
            ax.set_xlabel('Day of Month')
            ax.set_ylabel('Proportion of anomalies')
        return

    anomaly_counts = self.anomalies['DateTime'].dt.day.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.day.value_counts().sort_index()

    all_days = range(1, 32)
    anomaly_counts = anomaly_counts.reindex(all_days, fill_value=0)
    total_counts = total_counts.reindex(all_days, fill_value=0)

    normalized_counts = anomaly_counts.divide(total_counts).fillna(0)

    if ax is None:
      fig, ax = plt.subplots(figsize=(12, 6)) # Wider figure for days

    ax.bar(normalized_counts.index, normalized_counts.values)
    ax.set_xlabel('Day of Month')
    ax.set_ylabel('Proportion of Anomalies')
    ax.set_title('Anomaly Frequency by Day (Normalized)')
    ax.set_xticks(range(1, 32))
    ax.set_ylim(0, max(normalized_counts.max() * 1.1, 0.05))
    plt.tight_layout()


  # plots anomaly frequency per hour
  def anomalies_per_hour(self, ax=None): # Added ax parameter
    if self.anomalies.empty:
        print("No anomalies found to plot frequency by hour.")
        if ax:
            ax.text(0.5, 0.5, 'No anomalies found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Anomaly Frequency by Hour (Normalized)')
            ax.set_xticks(range(0, 24))
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Proportion of anomalies')
        return

    anomaly_counts = self.anomalies['DateTime'].dt.hour.value_counts().sort_index()
    total_counts = self.data['DateTime'].dt.hour.value_counts().sort_index()

    all_hours = range(0, 24)
    anomaly_counts = anomaly_counts.reindex(all_hours, fill_value=0)
    total_counts = total_counts.reindex(all_hours, fill_value=0)

    normalized_counts = anomaly_counts.divide(total_counts).fillna(0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(normalized_counts.index, normalized_counts.values)
    ax.set_xlabel('Hour of Day (0-23)')
    ax.set_ylabel('Proportion of Anomalies')
    ax.set_title('Anomaly Frequency by Hour (Normalized)')
    ax.set_xticks(range(0, 24))
    ax.set_ylim(0, max(normalized_counts.max() * 1.1, 0.05))
    plt.tight_layout()


  # plots four main plots and returns them as base64 strings
  def summary_plots(self):
    import io, base64
    plots = {}

    # Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    self.final_plot(ax=ax1)
    buf = io.BytesIO()
    fig1.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['scatter_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Anomalies per Month
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    self.anomalies_per_month(ax=ax2)
    buf = io.BytesIO()
    fig2.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['month_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Longest Anomalous Streak
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    self.longest_anomalous_streak(ax=ax3)
    buf = io.BytesIO()
    fig3.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['longest_anomalous_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig3)

    # Longest Non-anomalous Streak
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    self.longest_nonanomalous_streak(ax=ax4)
    buf = io.BytesIO()
    fig4.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['longest_clean_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig4)

    return plots