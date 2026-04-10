import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use Agg backend for headless environments
plt.switch_backend('Agg')

def generate_qml_plots():
    print("--- ⚛️ Starting NetSage-IDS QML Research Visualization ---")
    
    # Define paths
    BENCHMARK_PATH = "results/benchmark_results.json"
    OUTPUT_DIR = "results/plots/qml"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Benchmark Data
    if not os.path.exists(BENCHMARK_PATH):
        print(f"Error: Benchmark file not found at {BENCHMARK_PATH}")
        return
        
    with open(BENCHMARK_PATH, 'r') as f:
        data = json.load(f)
        
    models = data['models']
    class_names = data['meta']['class_names']
    
    # 1. Accuracy and F1 Comparison
    print("Plotting QML Accuracy & F1 Comparison...")
    model_names = [m['model'] for m in models]
    accuracies = [m['accuracy'] for m in models]
    f1_scores = [m['f1_macro'] for m in models]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#00ff88')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Macro', color='#7000ff')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Classical vs. Quantum Performance Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Quantum Latency Benchmarks
    print("Plotting Quantum Latency Benchmarks...")
    latencies = [m['inference_latency_ms'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, latencies, color='#ff0055')
    plt.yscale('log') # Log scale as latencies vary by orders of magnitude
    plt.ylabel('Inference Latency (ms) [Log Scale]')
    plt.title('The "Quantum Gap": Inference Latency Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f} ms', 
                 va='bottom', ha='center', fontsize=9)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_benchmarks.png'), dpi=300)
    plt.close()
    
    # 3. Quantum Confusion Matrices
    print("Plotting QML Confusion Matrices...")
    for m in models:
        if "XGBoost" in m['model']: continue # Focus on QML
        
        cm = np.array(m['confusion_matrix'])
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Research Matrix: {m["model"]}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        safe_name = m['model'].replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{safe_name}.png'), dpi=300)
        plt.close()
        
    print(f"--- ✅ QML research plots saved successfully in {OUTPUT_DIR} ---")

if __name__ == "__main__":
    generate_qml_plots()
