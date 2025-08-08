#!/usr/bin/env python3
"""
Post-training performance evaluation script
Analyzes training logs to compare sparse vs full training performance
"""

import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

def parse_log_file(log_path):
    """Parse training log file to extract metrics"""
    metrics = {
        'epoch_times': [],
        'iter_times': [],
        'throughputs': [],
        'total_samples': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_acc': 0
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse epoch metrics
            if 'epoch' in line and 'train/' in line:
                # Extract metrics using regex
                if 'train/epoch_time' in line:
                    match = re.search(r"'train/epoch_time': ([\d.]+)", line)
                    if match:
                        metrics['epoch_times'].append(float(match.group(1)))
                        
                if 'train/avg_iter_time' in line:
                    match = re.search(r"'train/avg_iter_time': ([\d.]+)", line)
                    if match:
                        metrics['iter_times'].append(float(match.group(1)))
                        
                if 'train/throughput_samples_per_sec' in line:
                    match = re.search(r"'train/throughput_samples_per_sec': ([\d.]+)", line)
                    if match:
                        metrics['throughputs'].append(float(match.group(1)))
                        
                if 'train/total_samples' in line:
                    match = re.search(r"'train/total_samples': ([\d.]+)", line)
                    if match:
                        metrics['total_samples'].append(float(match.group(1)))
                        
                if 'train/loss' in line:
                    match = re.search(r"'train/loss': ([\d.]+)", line)
                    if match:
                        metrics['train_loss'].append(float(match.group(1)))
                        
                if 'train/top1' in line:
                    match = re.search(r"'train/top1': ([\d.]+)", line)
                    if match:
                        metrics['train_acc'].append(float(match.group(1)))
                        
            # Parse validation accuracy
            if 'val/top1' in line:
                match = re.search(r"'val/top1': ([\d.]+)", line)
                if match:
                    metrics['val_acc'].append(float(match.group(1)))
                    
            # Parse best accuracy
            if 'New best acc' in line:
                match = re.search(r'New best acc.*?: ([\d.]+)', line)
                if match:
                    metrics['best_acc'] = max(metrics['best_acc'], float(match.group(1)))
    
    return metrics

def calculate_flops_efficiency(throughput, model_flops):
    """Calculate FLOPs efficiency (GFLOPs/sec)"""
    return (throughput * model_flops) / 1e9

def analyze_training_run(run_dir, model_flops=None):
    """Analyze a single training run"""
    log_file = None
    
    # Find log file
    for file in os.listdir(run_dir):
        if file.endswith('.log'):
            log_file = os.path.join(run_dir, file)
            break
    
    if not log_file:
        print(f"No log file found in {run_dir}")
        return None
        
    metrics = parse_log_file(log_file)
    
    # Calculate statistics
    stats = {
        'run_dir': run_dir,
        'avg_epoch_time': np.mean(metrics['epoch_times']) if metrics['epoch_times'] else 0,
        'std_epoch_time': np.std(metrics['epoch_times']) if metrics['epoch_times'] else 0,
        'avg_iter_time': np.mean(metrics['iter_times']) if metrics['iter_times'] else 0,
        'std_iter_time': np.std(metrics['iter_times']) if metrics['iter_times'] else 0,
        'avg_throughput': np.mean(metrics['throughputs']) if metrics['throughputs'] else 0,
        'std_throughput': np.std(metrics['throughputs']) if metrics['throughputs'] else 0,
        'total_training_time': sum(metrics['epoch_times']),
        'final_train_acc': metrics['train_acc'][-1] if metrics['train_acc'] else 0,
        'final_val_acc': metrics['val_acc'][-1] if metrics['val_acc'] else 0,
        'best_val_acc': metrics['best_acc'],
        'total_epochs': len(metrics['epoch_times']),
        'samples_per_epoch': metrics['total_samples'][0] if metrics['total_samples'] else 0
    }
    
    # Calculate FLOPs efficiency if model FLOPs provided
    if model_flops and stats['avg_throughput'] > 0:
        stats['flops_efficiency_gflops_per_sec'] = calculate_flops_efficiency(
            stats['avg_throughput'], model_flops
        )
    
    return stats

def compare_runs(runs_stats):
    """Compare multiple training runs"""
    if len(runs_stats) < 2:
        print("Need at least 2 runs to compare")
        return
        
    print("\n" + "="*80)
    print("TRAINING PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    headers = ["Metric", "Unit"] + [f"Run {i+1}" for i in range(len(runs_stats))] + ["Speedup"]
    
    metrics_to_compare = [
        ("Average Epoch Time", "sec", "avg_epoch_time"),
        ("Average Iter Time", "ms", "avg_iter_time", 1000),
        ("Throughput", "samples/sec", "avg_throughput"),
        ("Total Training Time", "sec", "total_training_time"),
        ("Final Train Accuracy", "%", "final_train_acc"),
        ("Final Val Accuracy", "%", "final_val_acc"),
        ("Best Val Accuracy", "%", "best_val_acc"),
    ]
    
    if "flops_efficiency_gflops_per_sec" in runs_stats[0]:
        metrics_to_compare.append(("FLOPs Efficiency", "GFLOP/s", "flops_efficiency_gflops_per_sec"))
    
    print(f"{'Metric':<25} {'Unit':<12}", end="")
    for i, stats in enumerate(runs_stats):
        run_name = os.path.basename(stats['run_dir'])
        print(f"{run_name:<20}", end="")
    print(f"{'Speedup':<12}")
    print("-" * (25 + 12 + 20 * len(runs_stats) + 12))
    
    for metric_info in metrics_to_compare:
        metric_name, unit, key = metric_info[:3]
        multiplier = metric_info[3] if len(metric_info) > 3 else 1
        
        print(f"{metric_name:<25} {unit:<12}", end="")
        
        values = []
        for stats in runs_stats:
            value = stats[key] * multiplier
            values.append(value)
            print(f"{value:<20.3f}", end="")
        
        # Calculate speedup (baseline is first run)
        if values[0] > 0 and key in ["avg_epoch_time", "avg_iter_time", "total_training_time"]:
            speedup = values[0] / values[1] if len(values) > 1 and values[1] > 0 else 1.0
            print(f"{speedup:<12.2f}x")
        elif key in ["avg_throughput", "flops_efficiency_gflops_per_sec"]:
            speedup = values[1] / values[0] if len(values) > 1 and values[0] > 0 else 1.0
            print(f"{speedup:<12.2f}x")
        else:
            print(f"{'N/A':<12}")
    
    # Memory efficiency analysis
    print("\n" + "="*50)
    print("SPARSE TRAINING EFFICIENCY ANALYSIS")
    print("="*50)
    
    if len(runs_stats) >= 2:
        sparse_stats = runs_stats[1]  # Assume second run is sparse
        full_stats = runs_stats[0]    # Assume first run is full
        
        print(f"Time Efficiency:")
        print(f"  - Epoch time reduction: {((full_stats['avg_epoch_time'] - sparse_stats['avg_epoch_time']) / full_stats['avg_epoch_time'] * 100):.1f}%")
        print(f"  - Total training speedup: {(full_stats['total_training_time'] / sparse_stats['total_training_time']):.2f}x")
        
        print(f"\nAccuracy Comparison:")
        print(f"  - Accuracy drop: {(full_stats['best_val_acc'] - sparse_stats['best_val_acc']):.2f}%")
        print(f"  - Accuracy retention: {(sparse_stats['best_val_acc'] / full_stats['best_val_acc'] * 100):.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate training performance")
    parser.add_argument("--runs_dir", default="runs/cifar10/resnet18", 
                       help="Directory containing training runs")
    parser.add_argument("--model_flops", type=float, 
                       help="Model FLOPs for efficiency calculation")
    parser.add_argument("--specific_runs", nargs="+", 
                       help="Specific run directories to compare")
    
    args = parser.parse_args()
    
    runs_stats = []
    
    if args.specific_runs:
        # Analyze specific runs
        for run_dir in args.specific_runs:
            if os.path.exists(run_dir):
                stats = analyze_training_run(run_dir, args.model_flops)
                if stats:
                    runs_stats.append(stats)
    else:
        # Analyze all runs in directory
        if os.path.exists(args.runs_dir):
            for run_name in os.listdir(args.runs_dir):
                run_path = os.path.join(args.runs_dir, run_name)
                if os.path.isdir(run_path):
                    stats = analyze_training_run(run_path, args.model_flops)
                    if stats:
                        runs_stats.append(stats)
    
    if not runs_stats:
        print("No valid training runs found")
        return
    
    # Sort by run name for consistent ordering
    runs_stats.sort(key=lambda x: x['run_dir'])
    
    # Print individual run statistics
    for i, stats in enumerate(runs_stats):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}: {os.path.basename(stats['run_dir'])}")
        print(f"{'='*60}")
        for key, value in stats.items():
            if key != 'run_dir':
                print(f"{key:<30}: {value:.4f}")
    
    # Compare runs if multiple
    if len(runs_stats) > 1:
        compare_runs(runs_stats)

if __name__ == "__main__":
    main()