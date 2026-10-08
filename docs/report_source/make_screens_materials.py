"""Screenshots of the study-material feature (Playwright + installed Edge)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5, color_scheme="light")
    page.goto(URL)
    page.wait_for_function("document.getElementById('status').textContent.includes('Generator')", timeout=300_000)
    page.wait_for_selector("#busy.hidden", state="attached", timeout=300_000)
    page.select_option("#sampleSelect", "CSE230001")
    page.click("#loadSample")
    page.wait_for_selector("#busy.hidden", state="attached", timeout=300_000)
    page.click("#analyzeBtn")
    page.wait_for_selector("#recCard:not(.hidden)", timeout=300_000)
    page.wait_for_selector("#busy.hidden", state="attached", timeout=300_000)
    page.click("#viewPdf")
    page.wait_for_selector("#pdfViewer:not(.hidden)", timeout=120_000)
    page.wait_for_timeout(3000)
    page.evaluate("document.getElementById('recs').style.display='none'")
    page.locator("#recCard").screenshot(path=str(OUT / "ss11_materials.png"))
    print("saved ss11_materials.png")
    browser.close()
