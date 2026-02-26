#!/usr/bin/env python3
"""
Quick script to identify true noise features from existing masking results
"""

import json
import sys
import os

def analyze_masking_results(results_file, baseline_mae=None):
    """Analyze masking results to identify true noise"""
    
    if not os.path.exists(results_file):
        print(f"❌ File not found: {results_file}")
        return None
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Get baseline
    if baseline_mae is None:
        if 'baseline' in data:
            baseline_mae = data['baseline'].get('mae', data['baseline'].get('mae_unmasked'))
        elif 'mae_unmasked' in data:
            baseline_mae = data['mae_unmasked']
        elif 'baseline_mae' in data:
            baseline_mae = data['baseline_mae']
        else:
            print("⚠️  Could not find baseline MAE, using 71.615")
            baseline_mae = 71.615
    
    true_noise = []
    meaningful = []
    
    # Check individual results
    if 'individual_results' in data:
        for feat_idx, result in data['individual_results'].items():
            delta = result.get('delta_mae', result.get('mae_delta'))
            feat_idx_int = int(feat_idx) if isinstance(feat_idx, str) else feat_idx
            if delta < 0:
                true_noise.append((feat_idx_int, delta))
            else:
                meaningful.append((feat_idx_int, delta))
    
    # Check feature-specific results (e.g., "feature_1377_masked")
    for key, value in data.items():
        if isinstance(value, dict):
            # Check for feature_idx and mae_delta
            if 'feature_idx' in value and 'mae_delta' in value:
                feat_idx = value['feature_idx']
                delta = value['mae_delta']
                if delta < 0:
                    true_noise.append((feat_idx, delta))
                else:
                    meaningful.append((feat_idx, delta))
    
    # Sort
    true_noise_sorted = sorted(true_noise, key=lambda x: x[1])  # Most negative first
    meaningful_sorted = sorted(meaningful, key=lambda x: x[1], reverse=True)  # Most positive first
    
    print("="*60)
    print("TRUE NOISE IDENTIFICATION FROM MASKING RESULTS")
    print("="*60)
    print(f"\nBaseline MAE: {baseline_mae:.4f} months")
    print(f"\nTrue Noise Features (improve when masked): {len(true_noise_sorted)}")
    if true_noise_sorted:
        print("\nTop 20 True Noise Features (best improvement):")
        for i, (feat_idx, delta) in enumerate(true_noise_sorted[:20], 1):
            print(f"  {i:2d}. Feature {feat_idx:5d}: {delta:+.4f} months ({delta/baseline_mae*100:+.2f}%)")
    
    print(f"\nMeaningful Features (degrade when masked): {len(meaningful_sorted)}")
    if meaningful_sorted:
        print("\nTop 10 Most Meaningful Features (worst degradation):")
        for i, (feat_idx, delta) in enumerate(meaningful_sorted[:10], 1):
            print(f"  {i:2d}. Feature {feat_idx:5d}: {delta:+.4f} months ({delta/baseline_mae*100:+.2f}%)")
    
    return {
        'baseline_mae': baseline_mae,
        'true_noise_features': [x[0] for x in true_noise_sorted],
        'meaningful_features': [x[0] for x in meaningful_sorted],
        'true_noise_with_deltas': {str(x[0]): float(x[1]) for x in true_noise_sorted},
        'meaningful_with_deltas': {str(x[0]): float(x[1]) for x in meaningful_sorted}
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_existing_masking_results.py <results_file.json>")
        print("\nExample:")
        print("  python analyze_existing_masking_results.py linear_probe_results/masking_experiments_results.json")
        sys.exit(1)
    
    results = analyze_masking_results(sys.argv[1])
    
    if results:
        # Save
        output_file = sys.argv[1].replace('.json', '_true_noise.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        print(f"\n✓ Use true_noise_features for masking experiments!")


