#!/usr/bin/env python3
"""
Comprehensive analysis script for IMU dataset.
Analyzes duration, sample statistics, acceleration/velocity metrics, and more.
Outputs results to a markdown report.
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

def load_csv_file(filepath):
    """Load a CSV file and return data as list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return data

def parse_numeric(value):
    """Safely convert string to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def calculate_duration(data):
    """Calculate duration in seconds from timestamp data."""
    if len(data) < 2:
        return 0
    try:
        start = parse_numeric(data[0]['timestamp'])
        end = parse_numeric(data[-1]['timestamp'])
        if start is not None and end is not None:
            return (end - start) / 1000.0  # Convert from ms to seconds
    except:
        pass
    return 0

def calculate_stats(values):
    """Calculate min, max, mean, std dev for a list of numeric values."""
    values = [v for v in values if v is not None]
    if not values:
        return {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A', 'std': 'N/A'}
    
    min_val = min(values)
    max_val = max(values)
    mean_val = mean(values)
    std_val = stdev(values) if len(values) > 1 else 0
    
    return {
        'min': f"{min_val:.3f}",
        'max': f"{max_val:.3f}",
        'mean': f"{mean_val:.3f}",
        'std': f"{std_val:.3f}"
    }

def analyze_file(filepath):
    """Analyze a single CSV file."""
    data = load_csv_file(filepath)
    if not data:
        return None
    
    duration = calculate_duration(data)
    sample_count = len(data)
    
    # Extract numeric columns
    acc_x = [parse_numeric(row.get('acc_x', '')) for row in data]
    acc_y = [parse_numeric(row.get('acc_y', '')) for row in data]
    acc_z = [parse_numeric(row.get('acc_z', '')) for row in data]
    
    # Verify sample rate (expect ~100 Hz = 10ms intervals)
    timestamps = [parse_numeric(row.get('timestamp', '')) for row in data if parse_numeric(row.get('timestamp', '')) is not None]
    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i-1])
    avg_interval = mean(intervals) if intervals else 0
    sample_rate = 1000 / avg_interval if avg_interval > 0 else 0
    
    analysis = {
        'filepath': filepath,
        'sample_count': sample_count,
        'duration_sec': f"{duration:.3f}",
        'avg_sample_interval_ms': f"{avg_interval:.2f}",
        'sample_rate_hz': f"{sample_rate:.1f}",
        'acc_x': calculate_stats(acc_x),
        'acc_y': calculate_stats(acc_y),
        'acc_z': calculate_stats(acc_z),
    }
    
    return analysis

def parse_filename(filename):
    """Extract event type, subject, and location from filename."""
    # Format: 20260405T221716Z_subject1-neck.csv
    parts = filename.replace('.csv', '').split('_')
    if len(parts) >= 2:
        location_part = parts[1]  # e.g., "subject1-neck"
        if '-' in location_part:
            subject, location = location_part.rsplit('-', 1)
            return subject, location
    return None, None

def main():
    # Check for command-line argument first
    if len(sys.argv) > 1:
        recordings_dir = Path(sys.argv[1])
    else:
        # Open folder selection dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)  # Bring to front
        
        recordings_dir = filedialog.askdirectory(
            title="Select the data folder to analyze",
            initialdir=str(Path(__file__).parent.parent / 'recordings')
        )
        
        root.destroy()
        
        if not recordings_dir:
            print("No folder selected. Exiting.")
            return
        
        recordings_dir = Path(recordings_dir)
    
    # Verify directory exists
    if not recordings_dir.exists() or not recordings_dir.is_dir():
        print(f"Error: Directory does not exist: {recordings_dir}")
        sys.exit(1)
    
    results = defaultdict(list)
    
    # Walk through all subdirectories and find CSV files
    csv_files_found = 0
    for subdir in sorted(recordings_dir.iterdir()):
        if subdir.is_dir():
            # Look for CSV files in subdirectories
            for csv_file in sorted(subdir.glob('*.csv')):
                analysis = analyze_file(str(csv_file))
                if analysis:
                    subject, location = parse_filename(csv_file.name)
                    analysis['subject'] = subject
                    analysis['location'] = location
                    results[subdir.name].append(analysis)
                    csv_files_found += 1
        elif subdir.suffix == '.csv':
            # Also look for CSV files directly in the root folder
            analysis = analyze_file(str(subdir))
            if analysis:
                subject, location = parse_filename(subdir.name)
                analysis['subject'] = subject
                analysis['location'] = location
                results['root'].append(analysis)
                csv_files_found += 1
    
    if csv_files_found == 0:
        print(f"No CSV files found in: {recordings_dir}")
        sys.exit(1)
    
    # Generate markdown report
    report = generate_markdown_report(results, recordings_dir)
    
    # Save report in the parent directory of the data folder
    output_path = recordings_dir.parent / 'analysis_report.md'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis complete!")
    print(f"Found {csv_files_found} CSV files")
    print(f"Report saved to: {output_path}")

def generate_markdown_report(results, recordings_dir):
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# IMU Dataset Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Dataset Location: {recordings_dir}\n\n")
    
    # Summary statistics
    all_files = [f for event_files in results.values() for f in event_files]
    durations = [float(f['duration_sec']) for f in all_files]
    sample_counts = [f['sample_count'] for f in all_files]
    
    report.append("## Summary Statistics\n")
    report.append(f"- Total events recorded: {len(all_files)}\n")
    report.append(f"- Total event types: {len(results)}\n")
    report.append(f"- Duration range: {min(durations):.3f}s - {max(durations):.3f}s\n")
    report.append(f"- Average duration: {mean(durations):.3f}s\n")
    report.append(f"- Sample count range: {min(sample_counts)} - {max(sample_counts)}\n")
    report.append(f"- Average samples per file: {mean(sample_counts):.0f}\n\n")
    
    # Per-event analysis
    report.append("## Analysis by Event Type\n\n")
    
    for event_name in sorted(results.keys()):
        files = results[event_name]
        report.append(f"### {event_name.replace('_', ' ').title()}\n")
        report.append(f"**Files:** {len(files)}\n\n")
        
        # Duration stats for this event
        event_durations = [float(f['duration_sec']) for f in files]
        report.append("#### Duration Statistics\n")
        report.append(f"- Shortest: {min(event_durations):.3f}s\n")
        report.append(f"- Longest: {max(event_durations):.3f}s\n")
        report.append(f"- Average: {mean(event_durations):.3f}s\n")
        if len(event_durations) > 1:
            report.append(f"- Std Dev: {stdev(event_durations):.3f}s\n")
        report.append("\n")
        
        # Sample rate verification
        event_sample_rates = [float(f['sample_rate_hz']) for f in files]
        report.append("#### Sample Rate Verification\n")
        report.append(f"- Average: {mean(event_sample_rates):.1f} Hz\n")
        report.append(f"- Range: {min(event_sample_rates):.1f} - {max(event_sample_rates):.1f} Hz\n\n")
        
        # Acceleration statistics
        report.append("#### Acceleration Statistics (All axes combined)\n")
        all_acc_x = []
        all_acc_y = []
        all_acc_z = []
        for f in files:
            all_acc_x.extend([parse_numeric(v) for v in [f['acc_x']['min'], f['acc_x']['max'], f['acc_x']['mean']]])
            all_acc_y.extend([parse_numeric(v) for v in [f['acc_y']['min'], f['acc_y']['max'], f['acc_y']['mean']]])
            all_acc_z.extend([parse_numeric(v) for v in [f['acc_z']['min'], f['acc_z']['max'], f['acc_z']['mean']]])
        
        all_acc_x = [v for v in all_acc_x if v is not None]
        all_acc_y = [v for v in all_acc_y if v is not None]
        all_acc_z = [v for v in all_acc_z if v is not None]
        
        if all_acc_x:
            report.append(f"- Acc X: {min(all_acc_x):.3f} to {max(all_acc_x):.3f} (mean: {mean(all_acc_x):.3f})\n")
        if all_acc_y:
            report.append(f"- Acc Y: {min(all_acc_y):.3f} to {max(all_acc_y):.3f} (mean: {mean(all_acc_y):.3f})\n")
        if all_acc_z:
            report.append(f"- Acc Z: {min(all_acc_z):.3f} to {max(all_acc_z):.3f} (mean: {mean(all_acc_z):.3f})\n")
        report.append("\n")
        
        # Sensor location comparison
        by_location = defaultdict(list)
        by_subject = defaultdict(list)
        for f in files:
            if f['location']:
                by_location[f['location']].append(f)
            if f['subject']:
                by_subject[f['subject']].append(f)
        
        if len(by_location) > 1:
            report.append("#### Sensor Location Comparison\n")
            for location in sorted(by_location.keys()):
                location_files = by_location[location]
                loc_durations = [float(f['duration_sec']) for f in location_files]
                report.append(f"- **{location.title()}** ({len(location_files)} files): ")
                report.append(f"Avg duration: {mean(loc_durations):.3f}s\n")
            report.append("\n")
        
        if len(by_subject) > 1:
            report.append("#### Subject Comparison\n")
            for subject in sorted(by_subject.keys()):
                subject_files = by_subject[subject]
                subj_durations = [float(f['duration_sec']) for f in subject_files]
                report.append(f"- **{subject.title()}** ({len(subject_files)} files): ")
                report.append(f"Avg duration: {mean(subj_durations):.3f}s\n")
            report.append("\n")
        
        # File details table
        report.append("#### Individual File Details\n\n")
        report.append("| File | Subject | Location | Duration (s) | Samples | Sample Rate (Hz) |\n")
        report.append("|------|---------|----------|--------------|---------|------------------|\n")
        
        for f in files:
            filename = Path(f['filepath']).name
            report.append(f"| {filename} | {f['subject'] or 'N/A'} | {f['location'] or 'N/A'} | ")
            report.append(f"{f['duration_sec']} | {f['sample_count']} | {f['sample_rate_hz']} |\n")
        
        report.append("\n")
    
    return "".join(report)

if __name__ == '__main__':
    main()
