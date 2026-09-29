"""Real screenshots of the running web frontend (Playwright driving the installed Microsoft Edge)."""
import json
import random
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
BANK = Path(sys.argv[3])  # question_bank.json: lets the script act like a learner who studied
rng = random.Random(3)


def key(qid):
    for q in json.loads(BANK.read_text(encoding="utf-8")):
        if q["id"] == qid:
            return q["answer"]
    return 0


def shot(page, name, selector=None):
    target = page.locator(selector) if selector else page
    target.screenshot(path=str(OUT / name))
    print("saved", name)


def wait_idle(page):
    page.wait_for_selector("#busy.hidden", state="attached", timeout=300_000)
    page.wait_for_timeout(400)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5,
                            color_scheme="light")
    page.goto(URL)
    page.wait_for_function("document.getElementById('status').textContent.includes('Generator')", timeout=300_000)
    wait_idle(page)
    shot(page, "ss01_home.png")

    page.select_option("#sampleSelect", "CSE230001")
    page.click("#loadSample")
    wait_idle(page)
    shot(page, "ss02_loaded.png", "main > section.card >> nth=0")

    page.click(".tab[data-tab=upload]")
    shot(page, "ss03_upload_tab.png", "main > section.card >> nth=0")
    page.click(".tab[data-tab=sample]")

    page.click("#analyzeBtn")
    wait_idle(page)
    page.wait_for_selector("#recCard:not(.hidden)")
    shot(page, "ss04_state.png", "#stateCard")
    shot(page, "ss05_recommendations.png", "#recCard details.rec >> nth=0")

    page.fill("#rule", "3 and 67")
    page.click("#startEval")
    wait_idle(page)
    page.wait_for_selector("#evalArea .q")
    first = page.locator("#evalArea .concept-eval").first
    first.screenshot(path=str(OUT / "ss06_quiz.png"))
    print("saved ss06_quiz.png")

    # answer level by level like a learner who studied the plan: correct about 85% of the time
    graded = False
    for _ in range(60):
        submit = page.locator("#evalArea button.btn")
        if submit.count() == 0:
            break
        box = submit.first.locator("xpath=..")
        for q in box.locator(".q").all():
            radios = q.locator("input[type=radio]:not([disabled])")
            n = radios.count()
            if n:
                right = key(q.get_attribute("data-id"))
                pick = right if rng.random() < 0.85 else rng.choice([i for i in range(n) if i != right])
                radios.nth(pick).check()
        submit.first.click()
        wait_idle(page)
        if not graded:  # first graded level, answers still visible
            page.locator("#evalArea .concept-eval >> nth=0").locator("details").first.evaluate("d => d.open = true")
            shot(page, "ss07_graded.png", "#evalArea .concept-eval >> nth=0")
            graded = True
    shot(page, "ss08_eval_done.png", "#evalCard")

    page.click("#nextRound")
    wait_idle(page)
    page.wait_for_selector("#history:not(.hidden)")
    shot(page, "ss09_round2.png", "#stateCard")

    page.goto(URL + "docs")
    page.wait_for_selector(".opblock", timeout=60_000)
    page.wait_for_timeout(800)
    shot(page, "ss10_api_docs.png")
    browser.close()
