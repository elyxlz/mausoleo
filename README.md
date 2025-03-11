Il Messaggero Archive Scraper
This script uses Playwright to scrape pages from the Il Messaggero Archive on a daily basis. For each day, it navigates to the target URL, smoothly scrolls the viewer container to load all pages, captures blob URLs for each page image, and saves each page as a JPEG file. The output is organized hierarchically by year, month (using the full month name), and day.

Features
Date Range: Scrapes every day from January 1, 1880 to December 31, 2024.
Hierarchical Storage: Output directories are structured as:
php-template
Copy
data/<year>/<month_name>/<day>/
For example: data/1950/January/1/.
Resume Capability: If no resume date is provided, the script checks the data folder for the latest scraped date and resumes from the next day.
Headless Mode: Runs headless (no GUI) by default. Use the --headful flag to run with a GUI.
Progress Bar: Uses tqdm to display a progress bar for the entire scraping process.
Failure Handling: If scraping fails for 10 consecutive days, the program quits.
Prerequisites
Python 3.7+
Playwright:
Install with:
bash
Copy
pip install playwright
Then install the required browsers and dependencies:
bash
Copy
npx playwright install
npx playwright install-deps
tqdm:
Install with:
bash
Copy
pip install tqdm
Other required modules (e.g., datetime, logging, argparse) are part of the Python Standard Library.
Usage
Run the script from the command line. It accepts the following optional arguments:

--resume YYYY-MM-DD
Resume scraping from the specified date (format: YYYY-MM-DD). If not provided, the script will automatically check the data folder and resume from the day after the latest scraped date.

--headful
Run the script in headful (non-headless) mode. By default, it runs headless.

Examples
Run in default headless mode and auto-resume:

bash
Copy
python script.py
Run in headful mode:

bash
Copy
python script.py --headful
Resume from a specific date (e.g., January 1, 1950):

bash
Copy
python script.py --resume 1950-01-01
Logging & Output
Log File:
All logs are written to scrape.log in the script's directory.
Output Files:
The scraped pages are saved as JPEG files under the hierarchical directory structure. For example:
swift
Copy
data/1950/January/1/1.jpeg
data/1950/January/1/2.jpeg
...
Notes
Smooth Scrolling:
The script simulates a smooth scroll to load all pages. You can adjust the scroll parameters (steps and delay) if you notice pages being skipped.

Performance:
The script includes a short wait after scrolling (0.2 seconds) to ensure all blob URLs are captured. If you’re confident that the blobs are loaded immediately, you can reduce or remove this delay.

Headless Environment:
When running on a headless Linux server, make sure you install the necessary dependencies using:

bash
Copy
npx playwright install-deps
Consecutive Failures:
The program quits if it fails to scrape 10 consecutive days to prevent continuous errors.
