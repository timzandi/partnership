import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Generate Dummy Data ---
np.random.seed(42)
num_members = 250

# MemberQualityRank: 1 = best, 100 = worst, scale to 1-100
df = pd.DataFrame({
    'Gender': np.random.choice(['Male','Female'], num_members),
    'MemberQualityRank': np.random.randint(1,101,num_members),  # 1 to 100
    'SessionsPerWeek': np.random.normal(3,1,num_members),
    'Age': np.random.normal(35,10,num_members)
})

# Optional: simulate profit for overlay
df['Profit'] = (100 - df['MemberQualityRank'] + 1) * 5 - (df['SessionsPerWeek']*20 + np.random.normal(50,30,num_members))

# Convert rank to score: higher = better
df['MemberQualityScore'] = 100 - df['MemberQualityRank'] + 1

# Encode Gender
df['GenderEncoded'] = df['Gender'].map({'Male':0,'Female':1})

# Scale numeric features
numeric_features = df[['MemberQualityScore','SessionsPerWeek','Age']]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)

# Weight gender lightly
gender_weight = 0.3
gender_scaled = df['GenderEncoded'].values.reshape(-1,1) * gender_weight

# Combine for clustering
X = np.hstack([scaled_numeric, gender_scaled])

# KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- Plotting ---
plt.figure(figsize=(10,7))

shape_map = {'Male':'o','Female':'s'}
cluster_colors = ['#FF6B6B','#4ECDC4','#556270']

for gender in df['Gender'].unique():
    subset = df[df['Gender']==gender]
    plt.scatter(
        subset['MemberQualityScore'],
        subset['SessionsPerWeek'],
        s=150,
        c=[cluster_colors[i] for i in subset['Cluster']],
        marker=shape_map[gender],
        edgecolor='k',
        alpha=0.7,
        label=gender
    )

# Optional: highlight top/bottom 10% by quality score
top_cutoff = np.percentile(df['MemberQualityScore'], 90)
bottom_cutoff = np.percentile(df['MemberQualityScore'], 10)

for i,row in df.iterrows():
    if row['MemberQualityScore'] >= top_cutoff or row['MemberQualityScore'] <= bottom_cutoff:
        plt.scatter(
            row['MemberQualityScore'],
            row['SessionsPerWeek'],
            s=220,
            facecolors='none',
            edgecolors='black',
            linewidths=2
        )

plt.title('Natural Gym Member Clusters')
plt.xlabel('Member Quality Score (Higher = Better)')
plt.ylabel('Sessions per Week')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

# Sample data output
print(df.head(10))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Dummy Data (cyan cluster subset) ---
num_members = 200
df_cyan = pd.DataFrame({
    'MemberQualityScore': np.random.randint(1,101,num_members),
    'SessionsPerWeek': np.random.normal(3,1,num_members),
    'Age': np.random.normal(35,10,num_members),
    'GenderEncoded': np.random.choice([0,1],num_members),
})

# Top 10% label
df_cyan['Top10Percent'] = (df_cyan['MemberQualityScore'] >= 90).astype(int)

# Features and target
X = df_cyan[['MemberQualityScore','SessionsPerWeek','Age','GenderEncoded']]
y = df_cyan['Top10Percent']

# Scale numeric features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['MemberQualityScore','SessionsPerWeek','Age']] = scaler.fit_transform(X[['MemberQualityScore','SessionsPerWeek','Age']])

# Logistic Regression with balanced class weight
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_scaled, y)
y_prob = logreg.predict_proba(X_scaled)[:,1]

# Add predictions to df
df_cyan['PredProb'] = y_prob
df_cyan['PredClass'] = (y_prob >= 0.5).astype(int)

# --- Sort by predicted probability ---
df_cyan_sorted = df_cyan.sort_values(by='PredProb').reset_index(drop=True)

# --- Plot predicted probability vs actual 0/1 ---
plt.figure(figsize=(10,6))
plt.scatter(range(len(df_cyan_sorted)), df_cyan_sorted['Top10Percent'], c='black', label='Actual 0/1', alpha=0.6, s=60)
plt.scatter(range(len(df_cyan_sorted)), df_cyan_sorted['PredProb'], c='cyan', label='Predicted Probability', alpha=0.7, s=80)

plt.xlabel('Sorted Observation Index')
plt.ylabel('Top 10% Membership / Predicted Probability')
plt.title('Logistic Regression Predictions vs Actuals (Sorted by Probability)')
plt.legend()
plt.grid(True)
plt.show()

# --- Metrics across probability thresholds ---
thresholds = np.linspace(0.1,0.9,17)
accuracy_list, precision_list, recall_list, f1_list = [],[],[],[]

for t in thresholds:
    y_pred_t = (df_cyan_sorted['PredProb'] >= t).astype(int)
    accuracy_list.append(accuracy_score(df_cyan_sorted['Top10Percent'], y_pred_t))
    precision_list.append(precision_score(df_cyan_sorted['Top10Percent'], y_pred_t))
    recall_list.append(recall_score(df_cyan_sorted['Top10Percent'], y_pred_t))
    f1_list.append(f1_score(df_cyan_sorted['Top10Percent'], y_pred_t))

# --- Multiline chart for metrics ---
plt.figure(figsize=(10,6))
plt.plot(thresholds, accuracy_list, label='Accuracy', marker='o')
plt.plot(thresholds, precision_list, label='Precision', marker='s')
plt.plot(thresholds, recall_list, label='Recall', marker='^')
plt.plot(thresholds, f1_list, label='F1-Score', marker='d')
plt.xlabel('Probability Threshold')
plt.ylabel('Metric Value')
plt.title('Metrics vs Probability Threshold')
plt.legend()
plt.grid(True)
plt.show()
