import argparse
import asyncio
import base64
import datetime
import logging
import os
import re

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
BASE_URL = "https://archivio.ilmessaggero.it/sfoglio?" "testata=MG&edition=NAZIONALE&date={date}&page=1" "#page=1&zoom=page-fit&pagemode=none"

USER_DATA_DIR = "user_data"
AUTH_FILE = "auth.json"


def build_url(date_str: str) -> str:
    return BASE_URL.format(date=date_str)


def build_output_dir(day_str: str, data_dir: str) -> str:
    """
    Create an output directory with the following hierarchy:
      <data_dir>/<year>/<month_name>/<day>
    """
    date_obj = datetime.datetime.strptime(day_str, "%Y-%m-%d").date()
    year = str(date_obj.year)
    month = date_obj.strftime("%B")  # e.g., "January"
    day = str(date_obj.day)
    output_dir = os.path.join(data_dir, year, month, day)
    return output_dir


def get_unscraped_days(data_folder: str, start_date: datetime.date, end_date: datetime.date):
    """
    Return a sorted list of day strings (YYYY-MM-DD) for which either the output directory does not exist
    or exists but does not contain any JPEG files.
    """
    missing_days = []
    current_date = start_date
    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")
        output_dir = build_output_dir(day_str, data_folder)
        if not os.path.exists(output_dir) or not any(fname.endswith(".jpeg") for fname in os.listdir(output_dir)):
            missing_days.append(day_str)
        current_date += datetime.timedelta(days=1)
    return missing_days


async def extract_total_pages(page) -> int:
    """
    Extract the total number of pages from the span with id "numPages".
    Expected text format: "di 44" -> extracts 44.
    """
    locator = page.locator("#numPages")
    try:
        await locator.first.wait_for(timeout=15000)
    except PlaywrightTimeoutError as e:
        raise PlaywrightTimeoutError("Timeout waiting for #numPages element.") from e

    text = await locator.first.inner_text()
    logging.info("Found #numPages text: %s", text)
    m = re.search(r"di\s*(\d+)", text)
    if not m:
        raise ValueError("Regex did not match #numPages text.")
    total = int(m.group(1))
    logging.info("Total pages detected: %d", total)
    return total


async def smooth_scroll_container(page, steps=40, delay=80):
    """
    Smoothly scroll the container with ID 'viewerContainer'.
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


async def scrape_day_with_page(page, day_str: str, data_dir: str):
    url = build_url(day_str)
    logging.info("Scraping date %s: %s", day_str, url)
    output_dir = build_output_dir(day_str, data_dir)
    os.makedirs(output_dir, exist_ok=True)
    blob_urls = []

    def on_request(request):
        req_url = request.url
        if req_url.startswith("blob:") and req_url not in blob_urls:
            blob_urls.append(req_url)

    page.on("request", on_request)

    try:
        await page.goto(url)
        logging.info("Navigated to URL for day %s", day_str)
        await asyncio.sleep(5)

        try:
            await page.click(".sfoglio", timeout=5000)
            logging.info("Clicked viewer container for %s", day_str)
        except Exception as e:
            logging.error("Viewer container not clickable for %s: %s", day_str, e)

        try:
            await page.locator("#viewerContainer").wait_for(timeout=15000)
            logging.info("Viewer container present for %s", day_str)
        except Exception as e:
            raise Exception(f"Viewer container did not appear in time for {day_str}: {e}")

        total_pages = await extract_total_pages(page)
        logging.info("Total pages for %s: %d", day_str, total_pages)

        logging.info("Smoothly scrolling viewer container for %s", day_str)
        await smooth_scroll_container(page, steps=40, delay=80)

        logging.info("Final blob URLs captured for %s:", day_str)
        for url in blob_urls:
            logging.info(url)

        if not blob_urls:
            raise ValueError(f"No blob URLs captured for day {day_str}.")

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
                raise Exception(f"Failed to fetch blob data for {blob_url} on {day_str}")

            filename = os.path.join(output_dir, f"{index}.jpeg")
            with open(filename, "wb") as f:
                f.write(base64.b64decode(data_base64))
            logging.info("Saved blob to %s", filename)

    except Exception as e:
        logging.error("Error scraping %s: %s", day_str, e)
        raise e
    finally:
        logging.info("Finished scraping for %s", day_str)


async def get_browser_context(p, headless: bool, login_url: str):
    """
    Returns a browser context that is authenticated.
    If the auth file exists, it will be used.
    Otherwise, a persistent context is launched and navigates to the login_url (which should
    show the login page when not authenticated), waits for manual login, saves the auth state,
    and then re-launches a non-persistent context with the saved state.
    """
    # Additional flags for headless environments (e.g., no GUI)
    extra_args = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"] if headless else []

    if os.path.exists(AUTH_FILE):
        logging.info("Authentication file found. Using stored authentication state.")
        browser = await p.chromium.launch(headless=headless, args=extra_args)
        context = await browser.new_context(storage_state=AUTH_FILE)
        return context, browser
    else:
        logging.info("No authentication file found. Launching persistent context for manual login.")
        # For persistent context, include the extra args along with any custom args.
        context = await p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=headless,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/117.0.0.0 Safari/537.36"
            ),
            args=["--disable-blink-features=AutomationControlled"] + extra_args,
        )
        page = await context.new_page()
        await page.goto(login_url)
        logging.info(
            "Opened page %s. If not logged in, the login page should appear. Waiting 60 seconds for manual login...",
            login_url,
        )
        await asyncio.sleep(60)  # Adjust this wait time as needed for manual login.
        await context.storage_state(path=AUTH_FILE)
        logging.info("Authentication state saved to %s.", AUTH_FILE)
        await context.close()
        browser = await p.chromium.launch(headless=headless, args=extra_args)
        context = await browser.new_context(storage_state=AUTH_FILE)
        return context, browser


async def main():
    parser = argparse.ArgumentParser(description="Scrape Il Messaggero archives by day.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from date (YYYY-MM-DD)")
    parser.add_argument("--headful", action="store_true", help="Run in headful (non-headless) mode")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save scraped data",
    )
    args = parser.parse_args()

    headless = not args.headful
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    default_start_date = datetime.date(1880, 1, 1)
    end_date = datetime.date(2001, 1, 1)

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
        start_date = default_start_date

    missing_days = get_unscraped_days(data_dir, start_date, end_date)
    total_days = len(missing_days)
    if total_days == 0:
        logging.info("No days left to scrape.")
        return

    logging.info("Total days to scrape: %d", total_days)
    # Use the URL for the first missing day as the login URL.
    login_url = build_url(missing_days[0])

    async with async_playwright() as p:
        context, browser = await get_browser_context(p, headless, login_url)
        with tqdm(total=total_days, desc="Scraping days") as pbar:
            for day_str in missing_days:
                page = await context.new_page()
                try:
                    await scrape_day_with_page(page, day_str, data_dir)
                except Exception as e:
                    logging.error("Error scraping %s: %s", day_str, e)
                pbar.update(1)
                await page.close()
                await asyncio.sleep(1)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
