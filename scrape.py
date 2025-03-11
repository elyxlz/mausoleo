import argparse
import asyncio
import base64
import datetime
import logging
import os
import re
import sys

from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from playwright.async_api import (
    async_playwright,
)
from tqdm import tqdm

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scrape.log"), logging.StreamHandler()],
)

# Base URL with a placeholder for date.
BASE_URL = (
    "https://archivio.ilmessaggero.it/sfoglio?"
    "testata=MG&edition=NAZIONALE&date={date}&page=1"
    "#page=1&zoom=page-fit&pagemode=none"
)

USER_DATA_DIR = "user_data"


def build_url(date_str: str) -> str:
    return BASE_URL.format(date=date_str)


def build_output_dir(day_str: str, data_dir: str) -> str:
    """
    Create an output directory with the following hierarchy:
      <data_dir>/<year>/<month_name>/<day>
    where year and day are plain integers, and month_name is the full month name.
    """
    date_obj = datetime.datetime.strptime(day_str, "%Y-%m-%d").date()
    year = str(date_obj.year)
    month = date_obj.strftime("%B")  # e.g., "January"
    day = str(date_obj.day)
    output_dir = os.path.join(data_dir, year, month, day)
    return output_dir


def get_latest_scraped_date(data_folder: str) -> datetime.date:
    """
    Scan the data folder for the latest scraped date.
    The folder structure is assumed to be:
      <data_folder>/<year>/<month_name>/<day>
    Returns the latest date as a datetime.date object or None if none exists.
    """
    latest = None
    if not os.path.isdir(data_folder):
        return None

    for year in os.listdir(data_folder):
        year_path = os.path.join(data_folder, year)
        if os.path.isdir(year_path):
            try:
                year_int = int(year)
            except ValueError:
                continue
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    try:
                        month_num = datetime.datetime.strptime(month, "%B").month
                    except ValueError:
                        continue
                    for day in os.listdir(month_path):
                        day_path = os.path.join(month_path, day)
                        if os.path.isdir(day_path):
                            try:
                                day_int = int(day)
                                date_obj = datetime.date(year_int, month_num, day_int)
                                if latest is None or date_obj > latest:
                                    latest = date_obj
                            except Exception:
                                continue
    return latest


async def extract_total_pages(page) -> int:
    """
    Extract the total number of pages from the span with id "numPages".
    Expected text format: "di 44" -> extracts 44.
    """
    try:
        locator = page.locator("#numPages")
        await locator.first.wait_for(timeout=15000)
        text = await locator.first.inner_text()
        logging.info("Found #numPages text: %s", text)
        m = re.search(r"di\s*(\d+)", text)
        if m:
            total = int(m.group(1))
            logging.info("Total pages detected: %d", total)
            return total
        else:
            logging.warning(
                "Regex did not match #numPages text. Defaulting total pages to 1."
            )
    except PlaywrightTimeoutError:
        logging.warning(
            "Timeout waiting for #numPages element. Defaulting total pages to 1."
        )
    except Exception as e:
        logging.error("Error while extracting total pages: %s", e)
    return 1


async def smooth_scroll_container(page, steps=40, delay=80):
    """
    Smoothly scroll the container with ID 'viewerContainer' from left to right
    in small steps with a delay (in ms) between each step.
    """
    await page.evaluate(
        f"""
        async () => {{
            const container = document.getElementById('viewerContainer');
            if (container) {{
                const total = container.scrollWidth;
                const step = total / {steps};
                for (let pos = 0; pos <= total; pos += step) {{
                    container.scrollLeft = pos;
                    await new Promise(r => setTimeout(r, {delay}));
                }}
            }}
        }}
        """
    )
    await asyncio.sleep(0.2)


async def scrape_day(day_str: str, headless: bool, data_dir: str):
    url = build_url(day_str)
    logging.info("Scraping date %s: %s", day_str, url)

    # Create hierarchical output directory for this date.
    output_dir = build_output_dir(day_str, data_dir)
    os.makedirs(output_dir, exist_ok=True)

    async with async_playwright() as p:
        browser_type = p.chromium
        context = await browser_type.launch_persistent_context(
            USER_DATA_DIR,
            headless=headless,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            ),
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = await context.new_page()

        blob_urls = []

        def on_request(request):
            req_url = request.url
            if req_url.startswith("blob:") and req_url not in blob_urls:
                blob_urls.append(req_url)

        page.on("request", on_request)

        try:
            await page.goto(url)
            logging.info(
                "Browser launched for day %s. Please log in manually if needed.",
                day_str,
            )
            await asyncio.sleep(5)

            try:
                await page.click(".sfoglio", timeout=5000)
                logging.info("Clicked on viewer container to activate.")
            except Exception as e:
                logging.error(
                    "Viewer container not found/clickable for %s: %s", day_str, e
                )

            try:
                await page.locator("#viewerContainer").wait_for(timeout=15000)
                logging.info("Viewer container is present for %s.", day_str)
            except Exception as e:
                logging.error(
                    "Viewer container did not appear in time for %s: %s", day_str, e
                )

            total_pages = await extract_total_pages(page)
            logging.info("Total pages for %s: %d", day_str, total_pages)

            logging.info(
                "Smoothly scrolling the viewer container to load all pages for %s...",
                day_str,
            )
            await smooth_scroll_container(page, steps=40, delay=80)

            logging.info("Final blob URLs captured for %s:", day_str)
            for url in blob_urls:
                logging.info(url)

            for index, blob_url in enumerate(blob_urls, start=1):
                data_base64 = await page.evaluate(f"""
                    async () => {{
                        try {{
                            const response = await fetch("{blob_url}");
                            const buffer = await response.arrayBuffer();
                            let binary = '';
                            const bytes = new Uint8Array(buffer);
                            for (let i = 0; i < bytes.byteLength; i++) {{
                                binary += String.fromCharCode(bytes[i]);
                            }}
                            return btoa(binary);
                        }} catch (err) {{
                            return null;
                        }}
                    }}
                """)
                if data_base64 is None:
                    logging.error(
                        "Failed to fetch blob data for %s on %s", blob_url, day_str
                    )
                    continue

                filename = os.path.join(output_dir, f"{index}.jpeg")
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(data_base64))
                logging.info("Saved blob to %s", filename)

        except Exception as e:
            logging.error("Error scraping %s: %s", day_str, e)
            raise e
        finally:
            await context.close()
            logging.info("Finished scraping for %s.\n", day_str)


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape Il Messaggero archives by day."
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--headful", action="store_true", help="Run in headful (non-headless) mode"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save scraped data",
    )
    args = parser.parse_args()

    # Default is headless mode.
    headless = not args.headful
    data_dir = args.data_dir

    # Ensure base data directory exists.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    default_start_date = datetime.date(1880, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    if args.resume:
        try:
            resume_date = datetime.datetime.strptime(args.resume, "%Y-%m-%d").date()
            if resume_date < default_start_date or resume_date > end_date:
                logging.error("Resume date is out of range. Using default start date.")
                start_date = default_start_date
            else:
                start_date = resume_date
        except Exception:
            logging.error("Invalid resume date format. Using default start date.")
            start_date = default_start_date
    else:
        latest = get_latest_scraped_date(data_dir)
        if latest:
            start_date = latest + datetime.timedelta(days=1)
            logging.info(
                "Resuming from next day after latest scraped date: %s", start_date
            )
        else:
            start_date = default_start_date

    total_days = (end_date - start_date).days + 1
    current_date = start_date

    consecutive_failures = 0

    with tqdm(total=total_days, desc="Scraping days") as pbar:
        while current_date <= end_date:
            day_str = current_date.strftime("%Y-%m-%d")
            try:
                await scrape_day(day_str, headless, data_dir)
                consecutive_failures = 0  # reset on success
            except Exception as e:
                logging.error("Error in main while scraping %s: %s", day_str, e)
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    logging.error("Failed 10 consecutive days. Quitting.")
                    sys.exit(1)
            current_date += datetime.timedelta(days=1)
            pbar.update(1)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
