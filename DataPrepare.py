import json
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

# Define normal distribution
def normal_dist(x, mu, sigma, amplitude):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def travel_time_dist(df_order, df_values, route_id):
  df_order = df_order[['start_stop_id', 'end_stop_id']].copy()
  df_order.columns = ['start_stop_id', 'end_stop_id']
  df_sorted = pd.merge(df_order, df_values, on=['start_stop_id', 'end_stop_id'], how='left')

  # 5. Calculate parameters of distribution
  df = df_sorted

  # Extract data
  data_columns = df.columns[3:]
  data = df[data_columns]
  df['mu'] = np.nan
  df['sigma'] = np.nan

  # fit each row
  for index, row in df.iterrows():
      row_data = row[data_columns].values
      non_zero_indices = np.where(row_data > 0)[0]

      if len(non_zero_indices) < 2:  # at least two data points
          print(f"Row {index} (from {row['start_stop_id']} to {row['end_stop_id']}): No enough data point")
          continue

      # Non-zero
      x_values = data_columns[non_zero_indices].astype(int)
      y_values = row_data[non_zero_indices]

      try:
          mu_guess = np.average(x_values, weights=y_values)
          sigma_guess = np.sqrt(np.average((x_values - mu_guess)**2, weights=y_values))
          amplitude_guess = y_values.max()

          # fit normal distribution
          popt, pcov = curve_fit(normal_dist, x_values, y_values,
                                p0=[mu_guess, sigma_guess, amplitude_guess])
          
          # add into
          df.loc[index, 'mu'] = popt[0] if popt[0] > 0 else None
          df.loc[index, 'sigma'] = popt[1] if popt[0] > 0 else None
          df.to_excel(f"/TravelTime/travel_time_norm_{route_id}.xlsx", index=False)

      except Exception as e:
          print(f"Row {index} (from {row['start_stop_id']} to {row['end_stop_id']}) fitting failure: {str(e)}, use the simpliest way in training")


ROUTE_ID = 'A2386'
# 1. Stops and segments in route
file = f'/Route/route_{ROUTE_ID}.json' # original data received 
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract
filename = os.path.basename(file)  # 'route_A2386.json'
route_id = filename.replace('route_', '').replace('.json', '')  # 'A2386'

results = []
for segment in data['segments']:
    results.append({
        'route_id': route_id,
        'start_stop_id': segment['start_stop_id'],
        'end_stop_id': segment['end_stop_id'],
        'length': segment['length']
    })

# save to CSV
save_dir = '/Route'
filename = f'route_segments_{ROUTE_ID}.csv'
save_path = os.path.join(save_dir, filename)

with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['route_id', 'start_stop_id', 'end_stop_id', 'length']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# 2. Historical data
path = '/TravelTime/hist_data.csv'  # original data received 
df = pd.read_csv(path)
df[['start_stop_id', 'end_stop_id']] = df['segment'].str.split('_', expand=True)

# Grouped by route_id, stop_id
grouped = df.groupby(['route_id', 'start_stop_id', 'end_stop_id'])
# Travel time distribution
prob_dist = (
    df.groupby(['route_id', 'start_stop_id', 'end_stop_id'])['h_travel_t']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index() 
)

for route_id in prob_dist['route_id'].unique():
    route_df = prob_dist[prob_dist['route_id'] == route_id]
    route_df.to_excel(f'/TravelTime/travel_time_{route_id}.xlsx', index=False)

# 4. Ordered by stops; Caculate distribution
df_order = pd.read_csv(f"/Route/route_segments_{ROUTE_ID}.csv")   # First file
df_values = pd.read_excel(f"/TravelTime/travel_time_{ROUTE_ID}.xlsx")  # Second file

travel_time_dist(df_order, df_values, ROUTE_ID)
