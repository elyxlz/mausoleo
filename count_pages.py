#!/usr/bin/env python3
import os
import datetime
import argparse
from collections import defaultdict
from pathlib import Path


def count_pages_by_day(base_dir):
    """
    Count pages for each day since 1880 in the given directory.
    
    Args:
        base_dir (str): The directory to search for files
        
    Returns:
        dict: A dictionary mapping (year, month, day) to page count
    """
    counts = defaultdict(int)
    
    for year_dir in sorted(os.listdir(base_dir)):
        year_path = os.path.join(base_dir, year_dir)
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            continue
            
        year = int(year_dir)
        
        # Skip if year is before 1880
        if year < 1880:
            continue
            
        for month_dir in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_dir)
            if not os.path.isdir(month_path):
                continue
                
            # Convert month name to number
            try:
                month = datetime.datetime.strptime(month_dir, "%B").month
            except ValueError:
                try:
                    month = int(month_dir)
                except ValueError:
                    continue
                    
            for day_dir in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day_dir)
                if not os.path.isdir(day_path):
                    continue
                    
                try:
                    day = int(day_dir)
                except ValueError:
                    continue
                    
                # Count files in the day directory
                page_count = 0
                for _, _, files in os.walk(day_path):
                    page_count += len([f for f in files if f.endswith(('.jpeg', '.jpg', '.png'))])
                
                counts[(year, month, day)] = page_count
    
    return counts


def generate_html(counts, output_file):
    """
    Generate an HTML report from the page counts.
    
    Args:
        counts (dict): Dictionary mapping (year, month, day) to page count
        output_file (str): Path to save the HTML file
    """
    # Get min and max years
    min_year = min(year for year, _, _ in counts.keys()) if counts else 1880
    max_year = max(year for year, _, _ in counts.keys()) if counts else datetime.datetime.now().year
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Page Count by Day Since 1880</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
        }
        .year-section {
            margin-bottom: 20px;
        }
        .year-header {
            background-color: #f0f0f0;
            padding: 5px;
            cursor: pointer;
            margin-bottom: 5px;
        }
        .month-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 5px;
            margin-bottom: 10px;
        }
        .month-header {
            font-weight: bold;
            text-align: center;
            background-color: #e0e0e0;
            padding: 5px;
        }
        .day-grid {
            display: grid;
            grid-template-columns: repeat(31, 1fr);
            gap: 2px;
        }
        .day-cell {
            text-align: center;
            padding: 3px;
            font-size: 12px;
        }
        .has-pages {
            background-color: #c8e6c9;
        }
        .no-pages {
            background-color: #ffcdd2;
        }
        .empty {
            background-color: #f5f5f5;
        }
        .info-panel {
            position: fixed;
            top: 10px;
            right: 10px;
            background-color: white;
            border: 1px solid #ccc;
            padding: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        summary {
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Page Count by Day Since 1880</h1>
    <div class="info-panel">
        <p>Total days with pages: <span id="days-with-pages">0</span></p>
        <p>Total pages: <span id="total-pages">0</span></p>
    </div>
"""

    total_pages = sum(counts.values())
    days_with_pages = len([count for count in counts.values() if count > 0])

    # Add years
    for year in range(min_year, max_year + 1):
        html += f"""
    <details class="year-section">
        <summary class="year-header">Year {year}</summary>
"""

        # Add month headers
        html += """        <div class="month-grid">"""
        for month in range(1, 13):
            month_name = datetime.date(2000, month, 1).strftime("%b")
            html += f"""
            <div class="month-header">{month_name}</div>"""
        html += """
        </div>"""

        # Add days for each month
        for month in range(1, 13):
            html += f"""
        <div class="month-grid">
            <div class="day-grid" style="grid-column: span 12;">"""
            
            days_in_month = 31
            if month in [4, 6, 9, 11]:
                days_in_month = 30
            elif month == 2:
                # Check if it's a leap year
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    days_in_month = 29
                else:
                    days_in_month = 28
            
            for day in range(1, 32):
                if day <= days_in_month:
                    page_count = counts.get((year, month, day), 0)
                    if page_count > 0:
                        css_class = "has-pages"
                        title = f"{page_count} pages"
                    else:
                        css_class = "no-pages"
                        title = "No pages"
                    
                    html += f"""
                <div class="day-cell {css_class}" title="{title}">{day}<br>{page_count}</div>"""
                else:
                    html += """
                <div class="day-cell empty"></div>"""
                    
            html += """
            </div>
        </div>"""
            
        html += """
    </details>"""

    html += f"""
    <script>
        document.getElementById('days-with-pages').textContent = '{days_with_pages}';
        document.getElementById('total-pages').textContent = '{total_pages}';
    </script>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Count pages by day since 1880 and generate HTML report')
    parser.add_argument('--dir', default='/media/sdr', help='Directory to search for files (default: /media/sdr)')
    parser.add_argument('--output', default='page_count.html', help='Output HTML file (default: page_count.html)')
    args = parser.parse_args()
    
    print(f"Searching directory: {args.dir}")
    counts = count_pages_by_day(args.dir)
    print(f"Found {sum(counts.values())} pages across {len(counts)} days")
    
    print(f"Generating HTML report to {args.output}")
    generate_html(counts, args.output)
    print("Done!")


if __name__ == "__main__":
    main()