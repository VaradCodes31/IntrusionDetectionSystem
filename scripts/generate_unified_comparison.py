import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Use Agg backend for headless environments
plt.switch_backend('Agg')

def generate_unified_comparison():
    print("--- 🏆 Generating Unified Project Model Comparison ---")
    
    # Define paths
    BENCHMARK_PATH = "results/benchmark_results.json"
    OUTPUT_DIR = "results/plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Benchmark Data
    if not os.path.exists(BENCHMARK_PATH):
        print(f"Error: Benchmark file not found at {BENCHMARK_PATH}")
        return
        
    with open(BENCHMARK_PATH, 'r') as f:
        data = json.load(f)
        
    models_data = data['models']
    
    # Extract metrics
    model_names = [m['model'] for m in models_data]
    accuracies = [m['accuracy'] for m in models_data]
    f1_scores = [m['f1_macro'] for m in models_data]
    precisions = [m['precision'] for m in models_data]
    
    # Plotting
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-muted') # Use a clean style
    
    plt.bar(x - width, accuracies, width, label='Accuracy', color='#3498db', alpha=0.9)
    plt.bar(x, f1_scores, width, label='F1 Macro', color='#2ecc71', alpha=0.9)
    plt.bar(x + width, precisions, width, label='Precision', color='#e74c3c', alpha=0.9)
    
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Detection Architecture', fontsize=12, fontweight='bold')
    plt.title('NetSage-IDS: Comprehensive Model Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, model_names, fontsize=10)
    plt.ylim(0, 110)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add grid for better precision reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate values
    def annotate(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Re-plotting with annotations
    bars1 = plt.bar(x - width, accuracies, width, color='#3498db', alpha=0.9)
    bars2 = plt.bar(x, f1_scores, width, color='#2ecc71', alpha=0.9)
    bars3 = plt.bar(x + width, precisions, width, color='#e74c3c', alpha=0.9)
    
    annotate(bars1)
    annotate(bars2)
    annotate(bars3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'project_model_comparison.png'), dpi=300)
    plt.close()
    
    print(f"--- ✅ Unified comparison saved successfully in {OUTPUT_DIR} ---")

if __name__ == "__main__":
    generate_unified_comparison()
