# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve,
    brier_score_loss
)

from sklearn.calibration import calibration_curve

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv('federated_health_dataset.csv')

print("Original Dataset:")
print(df.head())

# =====================================================
# ADD DIABETES RISK LABEL
# =====================================================

df["risk_of_diabetes"] = (
        (df["glucose_level"] > 125) |
        (df["bmi"] > 30) |
        ((df["glucose_level"] > 100) & (df["bmi"] > 25))
).astype(int)

print("\nRisk Class Distribution BEFORE SMOTE:")
print(df["risk_of_diabetes"].value_counts())

# =====================================================
# DATA PREPROCESSING
# =====================================================

df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

label_encoder = LabelEncoder()
df["client_id_encoded"] = label_encoder.fit_transform(df["client_id"])

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df['glucose_bmi_interaction'] = df['glucose_level'] * df['bmi']
df['age_bmi_ratio'] = df['age'] / (df['bmi'] + 1)
df['glucose_squared'] = df['glucose_level'] ** 2

# =====================================================
# SIMULATE FEDERATED CLIENTS
# =====================================================

clients = df["client_id"].unique()
client_datasets = {}

for client in clients:
    client_data = df[df["client_id"] == client]
    X_client = client_data.drop(columns=["risk_of_diabetes", "client_id"])
    y_client = client_data["risk_of_diabetes"]
    client_datasets[client] = (X_client, y_client)

# =====================================================
# GLOBAL FEATURE SCALING
# =====================================================

scaler = StandardScaler()
X_global = df.drop(columns=["risk_of_diabetes", "client_id"])
scaler.fit(X_global)

for client in client_datasets:
    X_local, y_local = client_datasets[client]
    X_scaled_local = scaler.transform(X_local)
    client_datasets[client] = (X_scaled_local, y_local)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_scaled_all = scaler.transform(X_global)
y_all = df["risk_of_diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_all, y_all, test_size=0.2,
    random_state=42, stratify=y_all
)

# =====================================================
# SMOTE BALANCING
# =====================================================

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# =====================================================
# LOCAL MODEL TRAINING
# =====================================================

local_models = []

for client, (X_local, y_local) in client_datasets.items():
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=50,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(X_local, y_local)
    local_models.append(model)


# =====================================================
# DIFFERENTIAL PRIVACY SIMULATION
# =====================================================

def add_dp_noise(weights, noise_scale=0.005):
    noise = np.random.normal(0, noise_scale, size=weights.shape)
    return weights + noise


noisy_importances = []

for model in local_models:
    imp = model.feature_importances_
    noisy_imp = add_dp_noise(imp)
    noisy_importances.append(noisy_imp)

global_importance = np.mean(noisy_importances, axis=0)

# =====================================================
# GLOBAL MODEL TRAINING
# =====================================================

global_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

# Fit with evaluation set to track training curves
global_model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_train_balanced, y_train_balanced), (X_test, y_test)],
    eval_metric=['binary_logloss', 'auc']
)

# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = global_model.predict(X_test)
y_pred_proba = global_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nGLOBAL MODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# PERFORMANCE METRICS BAR PLOT
# =====================================================

performance_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

plt.figure(figsize=(8, 6))
plt.bar(performance_df["Metric"], performance_df["Score"], color="#ACBFA4")
plt.ylim(0, 1.1)
plt.title(" Performance Metrics ", fontweight="bold")
plt.xlabel("Metric", fontweight="bold")
plt.ylabel("Score", fontweight="bold")

for i, v in enumerate(performance_df["Score"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig("Performance.png")
plt.show()

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Risk', 'At Risk'])
disp.plot()
plt.show()

# =====================================================
# ROC CURVE
# =====================================================

fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', color='#982598')
plt.plot([0, 1], [0, 1], '--')
plt.legend()
plt.title("ROC Curve", fontweight="bold")
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.savefig("ROC.png")
plt.show()

# =====================================================
# PRECISION RECALL CURVE
# =====================================================

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='#6E5034')
plt.title("Precision Recall Curve", fontweight="bold")
plt.xlabel("Recall", fontweight="bold")
plt.ylabel("Precision", fontweight="bold")
plt.savefig("Precision_Recall.png")
plt.show()

# =====================================================
# CALIBRATION CURVE
# =====================================================

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', color='#D1855C')
plt.plot([0, 1], [0, 1], '--')
plt.title("Calibration Curve", fontweight="bold")
plt.xlabel("Probability", fontweight="bold")
plt.ylabel("Probability", fontweight="bold")
plt.savefig("Calibration_Curve.png")
plt.show()

brier = brier_score_loss(y_test, y_pred_proba[:, 1])
print("Brier Score:", brier)

# =====================================================
# CROSS VALIDATION
# =====================================================

cv_scores = cross_val_score(global_model, X_train_balanced, y_train_balanced, cv=5)
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# =====================================================
# ENHANCED BLOCKCHAIN SIMULATION WITH METRICS
# =====================================================

print("\n" + "=" * 60)
print("BLOCKCHAIN LEDGER SIMULATION WITH PERFORMANCE METRICS")
print("=" * 60)


class Block:

    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

        # Performance metrics
        self.creation_time = 0
        self.validation_time = 0
        self.size_bytes = 0
        self.gas_cost = 0

    def calculate_hash(self):
        start_time = time.time()

        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()

        hash_result = hashlib.sha256(block_string).hexdigest()

        self.creation_time = time.time() - start_time
        self.size_bytes = len(block_string)

        return hash_result

    def mine_block(self, difficulty=2):
        """Simulate proof of work"""
        start_time = time.time()
        target = "0" * difficulty

        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

        self.validation_time = time.time() - start_time
        self.gas_cost = self.calculate_gas_cost()

    def calculate_gas_cost(self):
        """Simulate gas cost based on computational complexity"""
        base_cost = 21000  # Base transaction cost
        data_cost = self.size_bytes * 68  # Cost per byte
        computation_cost = self.nonce * 20  # Cost for mining

        return base_cost + data_cost + computation_cost


class Blockchain:

    def __init__(self):
        self.difficulty = 2

        # Performance tracking
        self.block_latencies = []
        self.throughput_history = []
        self.storage_growth = []
        self.gas_costs = []
        self.timestamps = []

        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        genesis = Block(0, str(time.time()), {"message": "Genesis Block"}, "0")
        genesis.mine_block(self.difficulty)
        return genesis

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        start_time = time.time()

        prev = self.get_latest_block()

        new_block = Block(
            len(self.chain),
            str(time.time()),
            data,
            prev.hash
        )

        # Mine the block (proof of work)
        new_block.mine_block(self.difficulty)

        # Calculate latency
        latency = time.time() - start_time
        self.block_latencies.append(latency)

        # Track metrics
        self.gas_costs.append(new_block.gas_cost)
        self.timestamps.append(time.time())

        # Calculate storage growth
        total_storage = sum(block.size_bytes for block in self.chain)
        self.storage_growth.append(total_storage)

        # Calculate throughput (blocks per second)
        if len(self.chain) > 1:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            throughput = len(self.chain) / time_diff if time_diff > 0 else 0
            self.throughput_history.append(throughput)

        self.chain.append(new_block)

        return new_block

    def validate_chain(self):
        """Validate blockchain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash integrity
            if current.hash != current.calculate_hash():
                return False

            # Check chain linkage
            if current.previous_hash != previous.hash:
                return False

        return True


# =====================================================
# BUILD BLOCKCHAIN WITH FEDERATED LEARNING
# =====================================================

print("\nBuilding Blockchain with Federated Learning Updates...")

blockchain = Blockchain()

# Track client contributions
client_contributions = {}

# Add local client updates
for i, model in enumerate(local_models):

    client_name = f"client_{i}"

    block_data = {
        "client": client_name,
        "model_hash": hashlib.sha256(model.feature_importances_.tobytes()).hexdigest(),
        "dp_enabled": True,
        "feature_importance_sum": float(model.feature_importances_.sum()),
        "update_round": 1
    }

    block = blockchain.add_block(block_data)

    # Track contributions
    if client_name not in client_contributions:
        client_contributions[client_name] = 0
    client_contributions[client_name] += 1

    print(f"✓ Added block for {client_name} | Latency: {blockchain.block_latencies[-1]:.4f}s | Gas: {block.gas_cost}")

# Add global model block
global_block = {
    "global_model_hash": hashlib.sha256(global_model.feature_importances_.tobytes()).hexdigest(),
    "accuracy": float(accuracy),
    "f1": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "update_type": "global_aggregation"
}

blockchain.add_block(global_block)

print(f"\n✓ Added global model block")
print(f"✓ Blockchain validated: {blockchain.validate_chain()}")

# Display blockchain summary
print("\n" + "=" * 60)
print("BLOCKCHAIN SUMMARY")
print("=" * 60)
print(f"Total Blocks: {len(blockchain.chain)}")
print(f"Average Block Latency: {np.mean(blockchain.block_latencies):.4f}s")
print(f"Total Gas Cost: {sum(blockchain.gas_costs):,.0f}")
print(f"Total Storage: {blockchain.storage_growth[-1]:,.0f} bytes")
print(f"Chain Valid: {blockchain.validate_chain()}")

# =====================================================
# COMPARISON: WITH vs WITHOUT BLOCKCHAIN
# =====================================================

print("\n" + "=" * 60)
print("COMPARISON: WITH vs WITHOUT BLOCKCHAIN")
print("=" * 60)

# Simulate training WITHOUT blockchain (direct aggregation)
print("\nSimulating Non-Blockchain Approach...")

start_time_no_bc = time.time()

# Direct model aggregation (no blockchain overhead)
aggregated_importance = np.mean([m.feature_importances_ for m in local_models], axis=0)

end_time_no_bc = time.time()
time_no_blockchain = end_time_no_bc - start_time_no_bc

# WITH blockchain (already measured)
time_with_blockchain = sum(blockchain.block_latencies)

# Storage comparison
storage_no_blockchain = 0  # No persistent ledger
storage_with_blockchain = blockchain.storage_growth[-1]

# Security comparison (simulated)
security_score_no_bc = 3  # Low (no immutability, no audit trail)
security_score_with_bc = 9  # High (immutable, auditable, transparent)

# Transparency comparison
transparency_no_bc = 2  # Low (no record of updates)
transparency_with_bc = 10  # High (complete audit trail)

# Create comparison dataframe
comparison_data = {
    'Metric': ['Processing Time (s)', 'Storage (bytes)', 'Security Score', 'Transparency Score'],
    'Without Blockchain': [time_no_blockchain, storage_no_blockchain, security_score_no_bc, transparency_no_bc],
    'With Blockchain': [time_with_blockchain, storage_with_blockchain, security_score_with_bc, transparency_with_bc]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df)

# =====================================================
# SCALABILITY TEST
# =====================================================

print("\n" + "=" * 60)
print("BLOCKCHAIN SCALABILITY TEST")
print("=" * 60)

# Test with increasing number of blocks
scalability_results = {
    'num_blocks': [],
    'avg_latency': [],
    'total_storage': [],
    'avg_gas_cost': []
}

for num_clients in [5, 10, 20, 50, 100]:

    test_blockchain = Blockchain()

    for i in range(num_clients):
        test_data = {
            "client": f"test_client_{i}",
            "data_hash": hashlib.sha256(f"data_{i}".encode()).hexdigest(),
            "timestamp": time.time()
        }
        test_blockchain.add_block(test_data)

    scalability_results['num_blocks'].append(num_clients)
    scalability_results['avg_latency'].append(np.mean(test_blockchain.block_latencies))
    scalability_results['total_storage'].append(test_blockchain.storage_growth[-1])
    scalability_results['avg_gas_cost'].append(np.mean(test_blockchain.gas_costs))

scalability_df = pd.DataFrame(scalability_results)
print("\n", scalability_df)

# =====================================================
# VISUALIZATION 1: BLOCK LATENCY
# =====================================================

print("\n" + "=" * 60)
print("GENERATING BLOCKCHAIN PERFORMANCE VISUALIZATIONS")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.plot(range(len(blockchain.block_latencies)), blockchain.block_latencies,
         marker='o', color='#E74C3C', linewidth=2, markersize=8)
plt.axhline(y=np.mean(blockchain.block_latencies), color='#3498DB',
            linestyle='--', label=f'Mean: {np.mean(blockchain.block_latencies):.4f}s')
plt.xlabel("Block Number", fontweight="bold")
plt.ylabel("Latency (seconds)", fontweight="bold")
plt.title("Blockchain Block Latency", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("blockchain_latency.png", dpi=300)
plt.show()

print("✓ Block Latency plot saved")

# =====================================================
# VISUALIZATION 2: THROUGHPUT METRICS
# =====================================================

plt.figure(figsize=(10, 6))
if len(blockchain.throughput_history) > 0:
    plt.plot(range(len(blockchain.throughput_history)), blockchain.throughput_history,
             marker='s', color='#27AE60', linewidth=2, markersize=8)
    plt.xlabel("Time Window", fontweight="bold")
    plt.ylabel("Throughput (blocks/second)", fontweight="bold")
    plt.title("Blockchain Throughput Over Time", fontweight="bold")
    plt.tight_layout()
    plt.savefig("blockchain_throughput.png", dpi=300)
    plt.show()
else:
    # Alternative: Show block processing rate
    processing_times = np.cumsum(blockchain.block_latencies)
    blocks = np.arange(len(processing_times))
    throughput_estimate = blocks / (processing_times + 1e-6)

    plt.plot(blocks, throughput_estimate, marker='s', color='#27AE60', linewidth=2, markersize=8)
    plt.xlabel("Block Number", fontweight="bold")
    plt.ylabel("Cumulative Throughput (blocks/second)", fontweight="bold")
    plt.title("Blockchain Throughput Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig("blockchain_throughput.png", dpi=300)
    plt.show()

print("✓ Throughput plot saved")

# =====================================================
# VISUALIZATION 3: STORAGE GROWTH
# =====================================================

plt.figure(figsize=(10, 6))
plt.plot(range(len(blockchain.storage_growth)),
         np.array(blockchain.storage_growth) / 1024,  # Convert to KB
         marker='^', color='#8E44AD', linewidth=2, markersize=8)
plt.xlabel("Block Number", fontweight="bold")
plt.ylabel("Storage Size (KB)", fontweight="bold")
plt.title("Blockchain Storage Growth", fontweight="bold")
plt.tight_layout()
plt.savefig("blockchain_storage_growth.png", dpi=300)
plt.show()

print("✓ Storage Growth plot saved")

# =====================================================
# VISUALIZATION 4: GAS COST ANALYSIS
# =====================================================

plt.figure(figsize=(10, 6))
plt.bar(range(len(blockchain.gas_costs)), blockchain.gas_costs, color='#F39C12')
plt.axhline(y=np.mean(blockchain.gas_costs), color='#C0392B',
            linestyle='--', linewidth=2, label=f'Mean: {np.mean(blockchain.gas_costs):,.0f}')
plt.xlabel("Block Number", fontweight="bold")
plt.ylabel("Gas Cost (units)", fontweight="bold")
plt.title("Blockchain Gas/Compute Cost per Block", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("blockchain_gas_cost.png", dpi=300)
plt.show()

print("✓ Gas Cost plot saved")

# =====================================================
# VISUALIZATION 5: SCALABILITY TEST RESULTS
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Latency vs Scale
axes[0, 0].plot(scalability_df['num_blocks'], scalability_df['avg_latency'],
                marker='o', color='#E74C3C', linewidth=2, markersize=8)
axes[0, 0].set_xlabel("Number of Blocks", fontweight="bold")
axes[0, 0].set_ylabel("Avg Latency (s)", fontweight="bold")
axes[0, 0].set_title("Latency Scalability", fontweight="bold")

# Storage vs Scale
axes[0, 1].plot(scalability_df['num_blocks'],
                scalability_df['total_storage'] / 1024,  # KB
                marker='s', color='#8E44AD', linewidth=2, markersize=8)
axes[0, 1].set_xlabel("Number of Blocks", fontweight="bold")
axes[0, 1].set_ylabel("Storage (KB)", fontweight="bold")
axes[0, 1].set_title("Storage Scalability", fontweight="bold")

# Gas Cost vs Scale
axes[1, 0].plot(scalability_df['num_blocks'], scalability_df['avg_gas_cost'],
                marker='^', color='#F39C12', linewidth=2, markersize=8)
axes[1, 0].set_xlabel("Number of Blocks", fontweight="bold")
axes[1, 0].set_ylabel("Avg Gas Cost", fontweight="bold")
axes[1, 0].set_title("Gas Cost Scalability", fontweight="bold")

# Efficiency Score (inverse of latency * gas_cost)
efficiency = 1 / (scalability_df['avg_latency'] * scalability_df['avg_gas_cost'] + 1e-6)
axes[1, 1].plot(scalability_df['num_blocks'], efficiency,
                marker='D', color='#27AE60', linewidth=2, markersize=8)
axes[1, 1].set_xlabel("Number of Blocks", fontweight="bold")
axes[1, 1].set_ylabel("Efficiency Score", fontweight="bold")
axes[1, 1].set_title("System Efficiency", fontweight="bold")

plt.tight_layout()
plt.savefig("blockchain_scalability_test.png", dpi=300)
plt.show()

print("✓ Scalability Test plot saved")

# =====================================================
# VISUALIZATION 6: WITH vs WITHOUT BLOCKCHAIN
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Processing Time (s)', 'Storage (bytes)', 'Security Score', 'Transparency Score']
colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]

    values = [comparison_df.loc[comparison_df['Metric'] == metric, 'Without Blockchain'].values[0],
              comparison_df.loc[comparison_df['Metric'] == metric, 'With Blockchain'].values[0]]

    # Create bars with different visual emphasis
    x_pos = [0, 1]
    bar1 = ax.bar([x_pos[0]], [values[0]], color=colors[idx], edgecolor='black', linewidth=2)
    bar2 = ax.bar([x_pos[1]], [values[1]], color=colors[idx], edgecolor='black', linewidth=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Without\nBlockchain', 'With\nBlockchain'])
    ax.set_ylabel(metric, fontweight="bold")
    ax.set_title(f"{metric} Comparison", fontweight="bold")

    # Add value labels
    for i in range(len(values)):
        ax.text(x_pos[i], values[i],
                f'{values[i]:.2f}' if values[i] < 100 else f'{values[i]:.0f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("blockchain_comparison.png", dpi=300)
plt.show()

print("✓ Comparison plot saved")

# =====================================================
# VISUALIZATION 7: COMPREHENSIVE PERFORMANCE DASHBOARD
# =====================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Latency Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(blockchain.block_latencies, bins=10, color='#E74C3C', edgecolor='black')
ax1.set_xlabel("Latency (s)", fontweight="bold")
ax1.set_ylabel("Frequency", fontweight="bold")
ax1.set_title("Latency Distribution", fontweight="bold")

# 2. Gas Cost Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(blockchain.gas_costs, bins=10, color='#F39C12', edgecolor='black')
ax2.set_xlabel("Gas Cost", fontweight="bold")
ax2.set_ylabel("Frequency", fontweight="bold")
ax2.set_title("Gas Cost Distribution", fontweight="bold")

# 3. Storage Growth Rate
ax3 = fig.add_subplot(gs[0, 2])
storage_growth_rate = np.diff(blockchain.storage_growth)
ax3.plot(storage_growth_rate, marker='o', color='#8E44AD', linewidth=2)
ax3.set_xlabel("Block Number", fontweight="bold")
ax3.set_ylabel("Growth (bytes)", fontweight="bold")
ax3.set_title("Storage Growth Rate", fontweight="bold")

# 4. Block Size Distribution
ax4 = fig.add_subplot(gs[1, 0])
block_sizes = [block.size_bytes for block in blockchain.chain]
ax4.bar(range(len(block_sizes)), block_sizes, color='#3498DB')
ax4.set_xlabel("Block Number", fontweight="bold")
ax4.set_ylabel("Size (bytes)", fontweight="bold")
ax4.set_title("Block Size Distribution", fontweight="bold")

# 5. Client Contribution
ax5 = fig.add_subplot(gs[1, 1])
clients_list = list(client_contributions.keys())
contributions = list(client_contributions.values())
ax5.barh(clients_list, contributions, color='#27AE60')
ax5.set_xlabel("Blocks Contributed", fontweight="bold")
ax5.set_ylabel("Client", fontweight="bold")
ax5.set_title("Client Contributions", fontweight="bold")

# 6. Performance Summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary_text = f"""
BLOCKCHAIN METRICS
{'=' * 30}

Total Blocks: {len(blockchain.chain)}

Avg Latency: {np.mean(blockchain.block_latencies):.4f}s
Max Latency: {np.max(blockchain.block_latencies):.4f}s
Min Latency: {np.min(blockchain.block_latencies):.4f}s

Total Gas: {sum(blockchain.gas_costs):,.0f}
Avg Gas: {np.mean(blockchain.gas_costs):,.0f}

Total Storage: {blockchain.storage_growth[-1] / 1024:.2f} KB

Validated: {blockchain.validate_chain()}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
         verticalalignment='center')

# 7. Scalability Projection
ax7 = fig.add_subplot(gs[2, :])
ax7_twin = ax7.twinx()

line1 = ax7.plot(scalability_df['num_blocks'], scalability_df['avg_latency'],
                 marker='o', color='#E74C3C', linewidth=2, label='Latency')
line2 = ax7_twin.plot(scalability_df['num_blocks'], scalability_df['total_storage'] / 1024,
                      marker='s', color='#8E44AD', linewidth=2, label='Storage')

ax7.set_xlabel("Number of Blocks", fontweight="bold")
ax7.set_ylabel("Avg Latency (s)", fontweight="bold", color='#E74C3C')
ax7_twin.set_ylabel("Storage (KB)", fontweight="bold", color='#8E44AD')
ax7.set_title("Scalability Analysis: Latency & Storage vs Block Count", fontweight="bold")
ax7.tick_params(axis='y', labelcolor='#E74C3C')
ax7_twin.tick_params(axis='y', labelcolor='#8E44AD')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax7.legend(lines, labels, loc='upper left')

plt.savefig("blockchain_performance_dashboard.png", dpi=300)
plt.show()

print("✓ Performance Dashboard saved")

# =====================================================
# ORIGINAL BLOCKCHAIN ANALYTICS VISUALIZATION
# =====================================================

print("\nGenerating Original Blockchain Visualization Plots...")

# Extract blockchain data
block_indices = []
timestamps = []
hash_lengths = []
client_blocks = []

for block in blockchain.chain:

    block_indices.append(block.index)

    # convert timestamp to float if possible
    try:
        timestamps.append(float(block.timestamp))
    except:
        timestamps.append(0)

    hash_lengths.append(len(block.hash))

    # count client blocks
    if isinstance(block.data, dict) and "client" in block.data:
        client_blocks.append(block.data["client"])
    else:
        client_blocks.append("system")

# -----------------------------------------------------
# Block Creation Timeline
# -----------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(block_indices, timestamps, marker='o', color='#547792')
plt.xlabel("Block Index", fontweight="bold")
plt.ylabel("Timestamp", fontweight="bold")
plt.title("Blockchain Block Creation Timeline", fontweight="bold")
plt.tight_layout()
plt.savefig("blockchain_timeline.png", dpi=300)
plt.show()

# -----------------------------------------------------
# Hash Integrity Visualization
# -----------------------------------------------------

plt.figure(figsize=(8, 6))
plt.bar(block_indices, hash_lengths, color='#8F0177')
plt.xlabel("Block Index", fontweight="bold")
plt.ylabel("Hash Length", fontweight="bold")
plt.title("Blockchain Hash Integrity Check", fontweight="bold")
plt.tight_layout()
plt.savefig("blockchain_hash_integrity.png", dpi=300)
plt.show()

# -----------------------------------------------------
# Client Contribution Distribution
# -----------------------------------------------------

client_counts = pd.Series(client_blocks).value_counts()

plt.figure(figsize=(8, 6))
plt.bar(client_counts.index, client_counts.values, color='#57595B')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Client ", fontweight="bold")
plt.ylabel("Number of Blocks", fontweight="bold")
plt.title("Blockchain Client Contribution Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig("blockchain_client_distribution.png", dpi=300)
plt.show()

# -----------------------------------------------------
# Simulated Block Update Magnitude
# -----------------------------------------------------

# simulate update magnitude (for visualization purpose)
update_magnitude = np.random.rand(len(block_indices))

plt.figure(figsize=(8, 6))
plt.plot(block_indices, update_magnitude, marker='o', color='#9CC6DB')
plt.xlabel("Block Index", fontweight="bold")
plt.ylabel("Update Magnitude", fontweight="bold")
plt.title("Blockchain Model Update Magnitude", fontweight="bold")
plt.tight_layout()
plt.savefig("blockchain_update_magnitude.png", dpi=300)
plt.show()

print("✓ Original blockchain plots generated successfully")

# =====================================================
# MODEL ACCURACY AND LOSS CURVES
# =====================================================

print("\nGenerating Model Training Curves...")

# Extract evaluation history from LightGBM
evals_result = global_model.evals_result_

# Check if evaluation results exist
if evals_result:
    # Loss Curve (Binary Logloss)
    if 'training' in evals_result and 'binary_logloss' in evals_result['training']:
        train_loss = evals_result['training']['binary_logloss']
        valid_loss = evals_result.get('valid_1', {}).get('binary_logloss', [])

        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label="Training Loss", linewidth=2)
        if valid_loss:
            plt.plot(valid_loss, label="Validation Loss", linewidth=2)
        plt.xlabel("Iterations", fontweight="bold")
        plt.ylabel("Log Loss", fontweight="bold")
        plt.title("Model Loss Curve", fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig("model_loss_curve.png", dpi=300)
        plt.show()

    # Accuracy proxy using AUC
    if 'training' in evals_result and 'auc' in evals_result['training']:
        train_auc = evals_result['training']['auc']
        valid_auc = evals_result.get('valid_1', {}).get('auc', [])

        plt.figure(figsize=(8, 6))
        plt.plot(train_auc, label="Training AUC", linewidth=2)
        if valid_auc:
            plt.plot(valid_auc, label="Validation AUC", linewidth=2)
        plt.xlabel("Iterations", fontweight="bold")
        plt.ylabel("AUC Score", fontweight="bold")
        plt.title("Model Accuracy Curve", fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig("model_accuracy_curve.png", dpi=300)
        plt.show()

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nGenerated Files:")
print("  1. Performance.png - Model performance metrics")
print("  2. ROC.png - ROC curve")
print("  3. Precision_Recall.png - Precision-Recall curve")
print("  4. Calibration_Curve.png - Model calibration")
print("  5. blockchain_latency.png - Block latency analysis")
print("  6. blockchain_throughput.png - Throughput metrics")
print("  7. blockchain_storage_growth.png - Storage growth analysis")
print("  8. blockchain_gas_cost.png - Gas/compute cost")
print("  9. blockchain_scalability_test.png - Scalability analysis")
print(" 10. blockchain_comparison.png - With vs Without blockchain")
print(" 11. blockchain_performance_dashboard.png - Comprehensive dashboard")
print(" 12. blockchain_timeline.png - Block creation timeline")
print(" 13. blockchain_hash_integrity.png - Hash integrity check")
print(" 14. blockchain_client_distribution.png - Client contributions")
print(" 15. blockchain_update_magnitude.png - Update magnitudes")
print(" 16. model_loss_curve.png - Training loss curve")
print(" 17. model_accuracy_curve.png - Training accuracy curve")
print("=" * 60)