"""
Strategy Chat – natural-language backtester for Nord Pool electricity spreads.

Supports complex instructions such as:
  "Buy dayahead sell intraday one for France and Germany, weekends,
   two strategies: hours 12-15 and hours 17-20, last one month"

Features:
  - Multiple areas in one command
  - Multiple hour ranges → side-by-side strategy comparison
  - Fuzzy matching for speech-to-text artifacts
  - Date range filtering (last N days/weeks/months)
  - Clarification form for missing parameters
  - Voice input via Web Speech API
"""

import io
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ── Data loading ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_MAP = {
    "dayahead": ("dayahead_prices.csv", "price"),
    "da":       ("dayahead_prices.csv", "price"),
    "day-ahead":("dayahead_prices.csv", "price"),
    "day ahead":("dayahead_prices.csv", "price"),
    "ida1":     ("ida1_prices.csv", "price"),
    "ida 1":    ("ida1_prices.csv", "price"),
    "intraday1":("ida1_prices.csv", "price"),
    "intraday 1":("ida1_prices.csv", "price"),
    "intraday one":("ida1_prices.csv", "price"),
    "intradayone":("ida1_prices.csv", "price"),
    "intraday auction one":("ida1_prices.csv", "price"),
    "intraday auction 1":("ida1_prices.csv", "price"),
    "ida2":     ("ida2_prices.csv", "price"),
    "ida 2":    ("ida2_prices.csv", "price"),
    "intraday2":("ida2_prices.csv", "price"),
    "intraday 2":("ida2_prices.csv", "price"),
    "intraday two":("ida2_prices.csv", "price"),
    "intraday auction two":("ida2_prices.csv", "price"),
    "intraday auction 2":("ida2_prices.csv", "price"),
    "ida3":     ("ida3_prices.csv", "price"),
    "ida 3":    ("ida3_prices.csv", "price"),
    "intraday3":("ida3_prices.csv", "price"),
    "intraday 3":("ida3_prices.csv", "price"),
    "intraday three":("ida3_prices.csv", "price"),
    "intraday auction three":("ida3_prices.csv", "price"),
    "intraday auction 3":("ida3_prices.csv", "price"),
    "vwap":     ("intraday_continuous_vwap_qh.csv", "vwap"),
    "continuous":("intraday_continuous_vwap_qh.csv", "vwap"),
}

# Speech-to-text fuzzy corrections
SPEECH_CORRECTIONS = {
    "their head": "dayahead",
    "they're head": "dayahead",
    "there head": "dayahead",
    "dare head": "dayahead",
    "day head": "dayahead",
    "day a head": "dayahead",
    "dead ahead": "dayahead",
    "they had": "dayahead",
    "intra day": "intraday",
    "inter day": "intraday",
    "in today": "intraday",
    "into day": "intraday",
    "intrude a": "intraday",
    "i d a 1": "ida1",
    "i d a 2": "ida2",
    "i d a 3": "ida3",
    "ida one": "ida1",
    "ida two": "ida2",
    "ida three": "ida3",
    "i da 1": "ida1",
    "i da 2": "ida2",
    "i da 3": "ida3",
    "vee wap": "vwap",
    "v wap": "vwap",
    "fee wap": "vwap",
}

DATASET_DISPLAY = {
    "dayahead_prices.csv": "DayAhead",
    "ida1_prices.csv": "IDA1",
    "ida2_prices.csv": "IDA2",
    "ida3_prices.csv": "IDA3",
    "intraday_continuous_vwap_qh.csv": "VWAP",
}

DATASET_OPTIONS = ["DayAhead", "IDA1", "IDA2", "IDA3", "VWAP"]
DATASET_KEY_MAP = {
    "DayAhead": "dayahead", "IDA1": "ida1", "IDA2": "ida2",
    "IDA3": "ida3", "VWAP": "vwap",
}

DAY_MAP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
    "weekdays": "weekdays", "weekday": "weekdays",
    "weekends": "weekends", "weekend": "weekends",
}

AREAS = ["AT", "BE", "FR", "GER", "NL"]
AREA_LABELS = {
    "AT": "AT (Austria)", "BE": "BE (Belgium)", "FR": "FR (France)",
    "GER": "GER (Germany)", "NL": "NL (Netherlands)",
}

# Country name → area code
COUNTRY_MAP = {
    "germany": "GER", "german": "GER",
    "france": "FR", "french": "FR",
    "belgium": "BE", "belgian": "BE",
    "austria": "AT", "austrian": "AT",
    "netherlands": "NL", "dutch": "NL",
    "holland": "NL",
}


@st.cache_data
def load_dataset(filename: str, price_col: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["deliveryStartCET"] = (
        pd.to_datetime(df["deliveryStartCET"], utc=True).dt.tz_convert("Europe/Paris")
    )
    df["hour"] = df["deliveryStartCET"].dt.hour
    df["day_of_week"] = df["deliveryStartCET"].dt.dayofweek
    df = df.rename(columns={price_col: "price_value"})
    # Preserve volume column for VWAP dataset
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df


# ── Voice input ─────────────────────────────────────────────────────────────
# Uses components.html() which renders inside an iframe with full <script> support.
# The component displays a mic button, live transcript, and a "Copy" fallback
# button. It also tries to push the transcript into the parent Streamlit chat
# input textarea via window.parent.document (works on same-origin localhost).

VOICE_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: transparent; padding: 8px 0; }
  #voice-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  #voiceBtn {
    background:#FF4B4B; color:white; border:none; border-radius:24px;
    padding:10px 20px; font-size:15px; cursor:pointer; display:flex;
    align-items:center; gap:6px; transition: background 0.2s;
  }
  #voiceBtn:hover { background:#e04343; }
  #voiceBtn.recording { background:#c0392b; animation: pulse 1.5s infinite; }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(192,57,43,0.4); }
    50%      { box-shadow: 0 0 0 10px rgba(192,57,43,0); }
  }
  #voiceStatus { color:#666; font-size:14px; max-width:400px;
                 overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #liveRow { display:flex; align-items:center; gap:8px; margin-top:6px; }
  #voiceLive { color:#333; font-size:13px; padding:6px 12px;
               background:#f0f2f6; border-radius:8px; min-height:24px;
               display:none; max-height:60px; overflow-y:auto;
               word-wrap:break-word; white-space:normal; flex:1; }
  #copyBtn { display:none; background:#4CAF50; color:white; border:none;
             border-radius:16px; padding:6px 14px; font-size:13px;
             cursor:pointer; white-space:nowrap; }
  #copyBtn:hover { background:#43a047; }
</style>
</head>
<body>
  <div id="voice-row">
    <button id="voiceBtn" onclick="toggleVoice()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" y1="19" x2="12" y2="23"/>
        <line x1="8" y1="23" x2="16" y2="23"/>
      </svg>
      <span id="btnLabel">Click to speak</span>
    </button>
    <span id="voiceStatus"></span>
  </div>
  <div id="liveRow">
    <div id="voiceLive"></div>
    <button id="copyBtn" onclick="copyText()">Copy</button>
  </div>

<script>
  var recognition = null;
  var isListening = false;
  var fullTranscript = '';

  function findParentChatInput() {
    try {
      var parentDoc = window.parent.document;
      var selectors = [
        'textarea[data-testid="stChatInputTextArea"]',
        'textarea[aria-label*="chat"]',
        'textarea[placeholder*="strategy"]',
        'textarea[placeholder*="e.g."]',
        '.stChatInput textarea',
        'div[data-testid="stChatInput"] textarea'
      ];
      for (var i = 0; i < selectors.length; i++) {
        var el = parentDoc.querySelector(selectors[i]);
        if (el) return el;
      }
      var all = parentDoc.querySelectorAll('textarea');
      for (var j = all.length - 1; j >= 0; j--) {
        if (all[j].offsetParent !== null) return all[j];
      }
    } catch(e) {
      // Cross-origin blocked — fall back to copy button
    }
    return null;
  }

  function pushToChat(text) {
    var chatInput = findParentChatInput();
    if (!chatInput) return false;
    try {
      var nativeSet = Object.getOwnPropertyDescriptor(
        window.parent.HTMLTextAreaElement.prototype, 'value'
      ).set;
      nativeSet.call(chatInput, text);
      chatInput.dispatchEvent(new Event('input', { bubbles: true }));
      chatInput.dispatchEvent(new Event('change', { bubbles: true }));
      chatInput.focus();
      return true;
    } catch(e) { return false; }
  }

  function copyText() {
    var text = fullTranscript.trim();
    if (!text) return;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(function() {
        document.getElementById('copyBtn').textContent = 'Copied!';
        setTimeout(function() { document.getElementById('copyBtn').textContent = 'Copy'; }, 1500);
      });
    } else {
      var ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta);
      ta.select(); document.execCommand('copy');
      document.body.removeChild(ta);
      document.getElementById('copyBtn').textContent = 'Copied!';
      setTimeout(function() { document.getElementById('copyBtn').textContent = 'Copy'; }, 1500);
    }
  }

  function toggleVoice() {
    var btn = document.getElementById('voiceBtn');
    var label = document.getElementById('btnLabel');
    var status = document.getElementById('voiceStatus');
    var live = document.getElementById('voiceLive');
    var copyBtnEl = document.getElementById('copyBtn');

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      status.textContent = 'Speech recognition not supported. Please use Chrome.';
      return;
    }

    if (isListening && recognition) {
      recognition.stop();
      return;
    }

    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    fullTranscript = '';

    recognition.onstart = function() {
      isListening = true;
      fullTranscript = '';
      label.textContent = 'Stop';
      btn.classList.add('recording');
      status.textContent = 'Listening...';
      live.style.display = 'block';
      live.textContent = '';
      copyBtnEl.style.display = 'none';
    };

    recognition.onresult = function(event) {
      var interim = '';
      var finalText = '';
      for (var i = 0; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalText += event.results[i][0].transcript + ' ';
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      fullTranscript = finalText.trim();
      var display = fullTranscript + (interim ? ' ' + interim : '');
      live.textContent = display || 'Listening...';
      live.style.display = 'block';
      if (display.trim()) pushToChat(display.trim());
    };

    recognition.onerror = function(event) {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        status.textContent = 'Microphone blocked. Allow mic access in browser settings.';
      } else if (event.error === 'no-speech') {
        status.textContent = 'No speech detected. Try again.';
      } else if (event.error === 'network') {
        status.textContent = 'Network error. Check internet connection.';
      } else {
        status.textContent = 'Error: ' + event.error;
      }
      isListening = false;
      label.textContent = 'Click to speak';
      btn.classList.remove('recording');
    };

    recognition.onend = function() {
      isListening = false;
      label.textContent = 'Click to speak';
      btn.classList.remove('recording');
      if (fullTranscript.trim()) {
        pushToChat(fullTranscript.trim());
        status.textContent = 'Done — paste into chat box and press Enter';
        live.textContent = fullTranscript.trim();
        copyBtnEl.style.display = 'inline-block';
      } else {
        status.textContent = '';
        live.style.display = 'none';
      }
    };

    try {
      recognition.start();
    } catch(e) {
      status.textContent = 'Could not start: ' + e.message;
    }
  }
</script>
</body>
</html>
"""


# ── Fuzzy text normalisation ────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Apply speech-to-text corrections and normalise the input."""
    result = text.lower()
    # Apply corrections longest-first to avoid partial replacements
    for wrong, right in sorted(SPEECH_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        # Use word-boundary regex to avoid cascading replacements
        pattern = r"(?<!\w)" + re.escape(wrong) + r"(?!\w)"
        result = re.sub(pattern, right, result)

    # ── Expand enumerated dataset references ──────────────────────
    # "intraday auction one two and three" → "intraday auction one intraday auction two intraday auction three"
    # "ida 1 2 and 3" → "ida1 ida2 ida3"
    # "intraday one, two and three" → "intraday one intraday two intraday three"
    number_words = {"one": "1", "two": "2", "three": "3", "1": "1", "2": "2", "3": "3"}

    # Pattern: (intraday [auction]?) followed by enumerated numbers/words
    m = re.search(
        r"(intraday\s+(?:auction\s+)?)"
        r"((?:(?:one|two|three|1|2|3)(?:\s*[,&]\s*|\s+and\s+|\s+)*)+)",
        result
    )
    if m:
        prefix = m.group(1).strip()  # e.g. "intraday auction" or "intraday"
        nums_text = m.group(2)
        nums = re.findall(r"(one|two|three|1|2|3)", nums_text)
        if len(nums) >= 2:
            expanded = " ".join(f"{prefix} {n}" for n in nums)
            result = result[:m.start()] + expanded + " " + result[m.end():]

    # Pattern: "ida" followed by enumerated numbers (ida 1 2 and 3)
    m = re.search(
        r"\bida\s+((?:(?:one|two|three|1|2|3)(?:\s*[,&]\s*|\s+and\s+|\s+)*)+)",
        result
    )
    if m:
        nums_text = m.group(1)
        nums = re.findall(r"(one|two|three|1|2|3)", nums_text)
        if len(nums) >= 2:
            expanded = " ".join(f"ida{number_words.get(n, n)}" for n in nums)
            result = result[:m.start()] + expanded + " " + result[m.end():]

    return result


# ── Rule-based parser (multi-area, multi-hour) ─────────────────────────────

def _find_area_at_position(text_lower: str, pos: int, direction: str = "after") -> Optional[str]:
    """Find the nearest area code/country name after or before a position in text."""
    best_area = None
    best_dist = 999

    # Build a list of (start_pos, end_pos, area_code) for all area mentions
    area_mentions = []
    for country, code in COUNTRY_MAP.items():
        for m in re.finditer(r"\b" + re.escape(country) + r"\b", text_lower):
            area_mentions.append((m.start(), m.end(), code))
    for a in AREAS:
        for m in re.finditer(r"\barea\s+" + re.escape(a.lower()) + r"\b", text_lower):
            area_mentions.append((m.start(), m.end(), a))
    for a in AREAS:
        if a in ("AT", "BE"):
            continue
        for m in re.finditer(r"(?:^|[\s,])" + re.escape(a.lower()) + r"(?:$|[\s,])", text_lower):
            # The actual area text starts after any leading space/comma
            actual_start = m.start()
            if actual_start < len(text_lower) and text_lower[actual_start] in " ,":
                actual_start += 1
            area_mentions.append((actual_start, m.end(), a))
    # Also check AT/BE with "area" prefix (already caught above) or in explicit patterns
    for a in ["AT", "BE"]:
        for m in re.finditer(r"\b" + re.escape(a.lower()) + r"\b", text_lower):
            area_mentions.append((m.start(), m.end(), a))

    for start, end, code in area_mentions:
        if direction == "after":
            dist = start - pos
            if dist >= 0 and dist < best_dist:
                best_dist = dist
                best_area = code
        else:  # "before"
            dist = pos - end
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best_area = code

    return best_area


def _parse_dataset_area_pairs(text_lower: str, datasets_found: list) -> list:
    """Parse dataset-area pairs from text like 'DA France vs IDA1 Germany'.

    Returns list of (dataset_key, area_code) tuples, or empty list if
    areas can't be unambiguously paired with datasets.
    """
    if len(datasets_found) < 2:
        return []

    pairs = []
    used_areas = set()

    for ds_key in datasets_found[:2]:  # Only first two datasets
        # Find position of this dataset in text
        # Try the exact key first
        m = re.search(r"\b" + re.escape(ds_key) + r"\b", text_lower)
        if not m:
            # Try as substring
            idx = text_lower.find(ds_key)
            if idx < 0:
                return []
            ds_end = idx + len(ds_key)
        else:
            ds_end = m.end()

        # Look for the nearest area AFTER the dataset mention
        area = _find_area_at_position(text_lower, ds_end, "after")
        if area and area not in used_areas:
            pairs.append((ds_key, area))
            used_areas.add(area)
        else:
            # Try before
            area = _find_area_at_position(text_lower, m.start() if m else idx, "before")
            if area and area not in used_areas:
                pairs.append((ds_key, area))
                used_areas.add(area)
            else:
                pairs.append((ds_key, None))

    return pairs


def _find_datasets_in_text(text_lower: str) -> list:
    """Find dataset names in text using word-boundary matching.

    Returns list of dataset keys IN ORDER OF APPEARANCE in the text,
    avoiding false positives like 'da' inside 'data' or 'today'.
    Uses word-boundary regex for short aliases.
    """
    # Collect (position, key) tuples
    hits = []
    seen_files = set()

    # First pass: try longer, unambiguous names (no boundary issues)
    for name in sorted(DATASET_MAP.keys(), key=len, reverse=True):
        if len(name) <= 2:
            continue  # skip short aliases like 'da' — handle below
        idx = text_lower.find(name)
        if idx >= 0:
            file_for_name = DATASET_MAP[name][0]
            if file_for_name not in seen_files:
                hits.append((idx, name))
                seen_files.add(file_for_name)

    # Second pass: short aliases (da, ida1, etc.) only with word boundaries
    for alias in ["da", "ida1", "ida2", "ida3", "vwap"]:
        m = re.search(r"\b" + re.escape(alias) + r"\b", text_lower)
        if m:
            file_for_alias = DATASET_MAP[alias][0]
            if file_for_alias not in seen_files:
                hits.append((m.start(), alias))
                seen_files.add(file_for_alias)

    # Sort by position in text so first-mentioned dataset is first
    hits.sort(key=lambda x: x[0])
    return [key for _, key in hits]


def detect_followup(text: str, current_strat: Optional[Dict]) -> Optional[Dict]:
    """Detect if user input is a follow-up modification of the current results.

    Returns a modified copy of current_strat if it's a follow-up, or None if
    it looks like a completely new query.

    Supported follow-ups:
      - Area filter: "only Germany", "filter France", "switch to NL"
      - Day filter: "weekends only", "filter weekdays", "only Monday"
      - Date zoom: "last 7 days", "zoom last week", "last 2 weeks"
      - Hour filter: "hours 8-16 only", "filter peak hours"
      - Dataset swap: "try IDA2 instead", "switch to VWAP", "use IDA1"
    """
    if current_strat is None:
        return None

    text_lower = normalise_text(text)

    # ── First: check if this looks like a NEW query ────────────────
    # If the text contains strong new-query signals, NEVER treat as follow-up.
    has_buy_sell = bool(re.search(r"\b(?:buy|sell|buying|selling)\b", text_lower))
    has_vs_compare = bool(re.search(r"\b(?:vs\.?|versus|compare|against)\b", text_lower))
    has_dataset = bool(_find_datasets_in_text(text_lower))
    has_lookup = bool(re.search(
        r"\b(?:show|pull|get|display|fetch|view|full|everything|prices?)\b", text_lower
    ))

    # If the text has buy/sell keywords → definitely a new query
    if has_buy_sell:
        return None
    # If the text has vs/compare with areas → new compare/cross query
    if has_vs_compare:
        return None
    # If the text mentions a dataset + lookup word → new lookup query
    if has_dataset and has_lookup:
        return None

    # Keywords that signal a follow-up modification
    followup_signals = bool(re.search(
        r"\b(?:only|filter|just|switch|change|try|instead|swap|zoom|narrow|focus|restrict|exclude|remove|add|also|include|with|without)\b",
        text_lower
    ))
    # Also treat short commands (< 5 words) as POTENTIAL follow-ups
    word_count = len(text_lower.split())
    is_short = word_count <= 4

    if not followup_signals and not is_short:
        return None

    import copy
    modified = copy.deepcopy(current_strat)
    changed = False

    # ── Area modifications ─────────────────────────────────────────
    new_areas = set()
    for country, code in COUNTRY_MAP.items():
        if re.search(r"\b" + re.escape(country) + r"\b", text_lower):
            new_areas.add(code)
    for a in AREAS:
        if a in ("AT", "BE"):
            continue
        if re.search(r"(?:^|[\s,])" + re.escape(a.lower()) + r"(?:$|[\s,])", text_lower):
            new_areas.add(a)
    if new_areas:
        # "exclude" / "remove" / "without" → remove areas
        if re.search(r"\b(?:exclude|remove|without|not|except)\b", text_lower):
            modified["areas"] = [a for a in modified.get("areas", AREAS) if a not in new_areas]
        # "add" / "also" / "include" → add areas
        elif re.search(r"\b(?:add|also|include|plus)\b", text_lower):
            existing = set(modified.get("areas", []))
            modified["areas"] = sorted(existing | new_areas)
        else:
            # "only X" / "filter X" / "switch to X" → replace areas
            modified["areas"] = sorted(new_areas)
        changed = True

    # ── Day modifications ──────────────────────────────────────────
    if re.search(r"\b(?:weekends?|saturday|sunday)\b", text_lower):
        if re.search(r"\b(?:exclude|remove|without|not|no)\b", text_lower):
            modified["days"] = [0, 1, 2, 3, 4]
        else:
            modified["days"] = [5, 6]
        changed = True
    elif re.search(r"\b(?:weekdays?|monday|tuesday|wednesday|thursday|friday)\b", text_lower):
        if re.search(r"\b(?:exclude|remove|without|not|no)\b", text_lower):
            modified["days"] = [5, 6]
        else:
            modified["days"] = [0, 1, 2, 3, 4]
        changed = True
    elif re.search(r"\ball\s+days?\b", text_lower):
        modified["days"] = None
        changed = True

    # ── Date range modifications ───────────────────────────────────
    date_match = re.search(
        r"(?:last|past|recent|zoom)\s+(?:to\s+)?(?:one\s+|1\s+)?(\d*)\s*(day|week|month|year)s?",
        text_lower)
    if date_match:
        num = int(date_match.group(1)) if date_match.group(1) else 1
        unit = date_match.group(2)
        if unit == "day":
            modified["date_filter_days"] = num
        elif unit == "week":
            modified["date_filter_days"] = num * 7
        elif unit == "month":
            modified["date_filter_days"] = num * 30
        elif unit == "year":
            modified["date_filter_days"] = num * 365
        changed = True

    # ── Hour range modifications ───────────────────────────────────
    hour_match = re.search(r"(?:hours?|between|from)\s*(\d{1,2})\s*(?:to|-|and)\s*(\d{1,2})", text_lower)
    if hour_match:
        h1, h2 = int(hour_match.group(1)), int(hour_match.group(2))
        modified["hour_ranges"] = [(min(h1, h2), max(h1, h2))]
        changed = True
    elif re.search(r"\b(?:peak)\b", text_lower) and not re.search(r"\boff-?peak\b", text_lower):
        modified["hour_ranges"] = [(8, 19)]
        changed = True
    elif re.search(r"\boff-?peak\b", text_lower):
        modified["hour_ranges"] = [(0, 7)]
        changed = True
    elif re.search(r"\ball\s+hours?\b", text_lower):
        modified["hour_ranges"] = []
        changed = True

    # ── Dataset swap (for lookup/compare modes) ────────────────────
    datasets_found = _find_datasets_in_text(text_lower)
    if datasets_found and re.search(r"\b(?:switch|swap|try|use|change|instead)\b", text_lower):
        new_ds = datasets_found[0]
        mode = modified.get("mode", "strategy")
        if mode == "lookup":
            modified["lookup_dataset"] = new_ds
        elif mode == "compare":
            modified["compare_dataset"] = new_ds
        elif mode == "timeshift":
            modified["dataset"] = new_ds
        elif mode in ("strategy", "cross", "multi_leg"):
            # Swap buy or sell based on context
            if re.search(r"\b(?:buy)\b", text_lower):
                modified["buy_dataset"] = new_ds
            elif re.search(r"\b(?:sell)\b", text_lower):
                modified["sell_dataset"] = new_ds
            else:
                # Default: swap buy dataset
                modified["buy_dataset"] = new_ds
        changed = True

    if changed:
        modified["missing"] = []
        return modified

    # If we got here with followup_signals but no change was made,
    # the user typed something like "change color to green" — recognized
    # as a follow-up intent but not a supported modification.
    # Return a special marker so the UI can show a helpful message.
    if followup_signals:
        return {"_unsupported_followup": True, "_text": text}

    return None


def _execute_and_store_followup(modified_strat: Dict):
    """Re-execute the modified strategy and store results in session state."""
    mode = modified_strat.get("mode", "strategy")

    if mode == "lookup":
        df, error = execute_lookup(modified_strat)
        if df is not None:
            lf = DATASET_MAP[modified_strat["lookup_dataset"]][0]
            ds_label = DATASET_DISPLAY.get(lf, modified_strat["lookup_dataset"])
            summary = f"Updated: **{len(df):,}** records for **{ds_label}**"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = (df, modified_strat)
            st.session_state.last_results_type = "lookup"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None
    elif mode == "compare":
        df, error = execute_compare(modified_strat)
        if df is not None:
            areas_str = " vs ".join(modified_strat["areas"])
            summary = f"Updated comparison: **{areas_str}** ({len(df):,} records)"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = (df, modified_strat)
            st.session_state.last_results_type = "compare"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None
    elif mode == "cross":
        df, error = execute_cross(modified_strat)
        if df is not None:
            summary = f"Updated cross-area ({len(df):,} slots)"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = (df, modified_strat)
            st.session_state.last_results_type = "cross"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None
    elif mode == "timeshift":
        df, error = execute_timeshift(modified_strat)
        if df is not None:
            summary = f"Updated time-shift ({len(df):,} days)"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = (df, modified_strat)
            st.session_state.last_results_type = "timeshift"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None
    elif mode == "multi_leg":
        df, error = execute_multi_leg(modified_strat)
        if df is not None:
            summary = f"Updated multi-leg ({len(df):,} slots)"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = (df, modified_strat)
            st.session_state.last_results_type = "multi_leg"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None
    else:
        results, error = execute_multi_strategy(modified_strat)
        if results is not None:
            total_slots = sum(len(df) for df, _ in results)
            summary = f"Updated strategy: **{len(results)}** variant(s), **{total_slots:,}** slots"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.last_results = results
            st.session_state.last_results_type = "strategy"
        else:
            st.session_state.messages.append({"role": "assistant", "content": error})
            st.session_state.last_results = None

    st.session_state.last_results_strat = modified_strat


def _detect_timeshift_pattern(text_lower: str) -> Optional[Dict]:
    """Detect time-shift pattern: buy HOURS sell HOURS on SAME dataset.

    Returns dict with keys: dataset, buy_hours, sell_hours, or None if not detected.
    Handles patterns like:
      - "buy first 4 hours sell last 4 hours DA Germany"
      - "buy hours 0-3 sell hours 20-23 IDA1 France"
      - "buy morning sell evening DA" (morning=6-11, evening=17-22)
      - "buy off-peak sell peak dayahead" (off-peak=0-7, peak=8-19)
    """
    # Check for explicit time-shift keywords
    has_buy = bool(re.search(r"\bbuy(?:ing)?\b", text_lower))
    has_sell = bool(re.search(r"\bsell(?:ing)?\b", text_lower))

    if not (has_buy and has_sell):
        return None

    # Find dataset mentioned
    datasets_found = _find_datasets_in_text(text_lower)
    if len(datasets_found) != 1:
        return None  # time-shift requires exactly one dataset

    dataset_key = datasets_found[0]

    # Define time windows
    time_windows = {
        "morning": (6, 11),
        "afternoon": (12, 16),
        "evening": (17, 22),
        "night": (23, 5),
        "peak": (8, 19),
        "off-peak": (0, 7), "offpeak": (0, 7),
    }

    result_dict = {"dataset": dataset_key, "buy_hours": None, "sell_hours": None}

    # Try to match explicit hour ranges: "buy hours 0-3 sell hours 20-23"
    # Require "hour/hours/first/last/at" keyword before number to avoid matching MW volumes
    buy_match = re.search(
        r"buy(?:ing)?\s+(?:(?:at\s+)?hours?\s+|(?:at\s+|first\s+|last\s+))(\d{1,2})(?:\s*[-to]+\s*(\d{1,2}))?",
        text_lower)
    sell_match = re.search(
        r"sell(?:ing)?\s+(?:(?:at\s+)?hours?\s+|(?:at\s+|first\s+|last\s+))(\d{1,2})(?:\s*[-to]+\s*(\d{1,2}))?",
        text_lower)

    if buy_match and sell_match:
        # Check if "first N hours" or "last N hours"
        first_match = re.search(r"buy(?:ing)?\s+first\s+(\d+)\s+hours?", text_lower)
        last_match = re.search(r"sell(?:ing)?\s+last\s+(\d+)\s+hours?", text_lower)

        if first_match:
            n = int(first_match.group(1))
            result_dict["buy_hours"] = (0, n - 1)
        elif buy_match.group(2):
            result_dict["buy_hours"] = (int(buy_match.group(1)), int(buy_match.group(2)))
        else:
            result_dict["buy_hours"] = (int(buy_match.group(1)), int(buy_match.group(1)))

        if last_match:
            n = int(last_match.group(1))
            result_dict["sell_hours"] = (24 - n, 23)
        elif sell_match.group(2):
            result_dict["sell_hours"] = (int(sell_match.group(1)), int(sell_match.group(2)))
        else:
            result_dict["sell_hours"] = (int(sell_match.group(1)), int(sell_match.group(1)))

        return result_dict if result_dict["buy_hours"] and result_dict["sell_hours"] else None

    # Try to match named windows: "buy morning sell evening"
    buy_window = None
    sell_window = None
    for window_name, (start, end) in time_windows.items():
        if re.search(r"\bbuy(?:ing)?\s+" + re.escape(window_name) + r"\b", text_lower):
            buy_window = (start, end) if start <= end else (start, 23)
        if re.search(r"\bsell(?:ing)?\s+" + re.escape(window_name) + r"\b", text_lower):
            sell_window = (start, end) if start <= end else (start, 23)

    if buy_window and sell_window:
        result_dict["buy_hours"] = buy_window
        result_dict["sell_hours"] = sell_window
        return result_dict

    return None


def _detect_multi_leg_pattern(text_lower: str) -> Optional[List[Dict]]:
    """Detect multi-leg volume pattern: buy XMW AREA, ... sell YMW AREA, ... on DATASET.

    Returns list of leg dicts: [{"side": "buy"/"sell", "volume_mw": float, "area": str}, ...]
    or None if no pattern detected.
    """
    # Pattern: NUMBER mw/megawatt AREA
    legs = []

    # Find all volume mentions with areas
    # Negative lookbehind prevents matching "ida1", "ida2", "ida3" as volume
    pattern = r"(?<![a-z])(\d+(?:\.\d+)?)\s*(?:mw|megawatt)?\s+([a-z]+)"

    for m in re.finditer(pattern, text_lower):
        volume_str = m.group(1)
        area_str = m.group(2).strip()

        # Try to match area
        area_code = None
        if area_str in AREAS:
            area_code = area_str.upper()
        elif area_str in COUNTRY_MAP:
            area_code = COUNTRY_MAP[area_str]

        if area_code:
            # Determine side (buy or sell) by looking before the number
            before_text = text_lower[:m.start()].lower()

            # Find the most recent buy/sell keyword
            last_buy = before_text.rfind("buy")
            last_sell = before_text.rfind("sell")

            if last_buy > last_sell:
                side = "buy"
            elif last_sell > last_buy:
                side = "sell"
            else:
                continue  # Skip if we can't determine side

            legs.append({
                "side": side,
                "volume_mw": float(volume_str),
                "area": area_code,
            })

    if len(legs) >= 2:
        return legs
    return None


def parse_instruction(text: str) -> Dict:
    """Parse a natural-language trading instruction OR data lookup request.

    Returns a dict supporting:
      - mode: "strategy" or "lookup"
      - areas: list of area codes (not just one)
      - hour_ranges: list of (start, end) tuples
      - date_filter_days: int or None (e.g. 30 for 'last month')
      - lookup_dataset: str or None (for lookup mode)

    Intelligence rules:
      - "full data", "all data", "everything", "give me data" → lookup mode
      - "compare X and Y", "X vs Y" → strategy (buy X, sell Y)
      - "all areas" → explicit all areas (NOT missing)
      - Areas are NEVER required — empty means "all areas"
      - Only buy_dataset/sell_dataset are truly required for strategy mode
    """
    raw = text
    text_lower = normalise_text(text)

    result = {
        "mode": "strategy",    # "strategy", "lookup", "compare", "cross", "timeshift", or "multi_leg"
        "buy_dataset": None,
        "sell_dataset": None,
        "lookup_dataset": None,
        "compare_dataset": None,
        "areas": [],           # list of area codes
        "hour_ranges": [],     # list of (start, end)
        "days": None,
        "date_filter_days": None,
        "missing": [],
        "raw": raw,
    }

    # ── Detect timeshift and multi-leg modes FIRST (highest priority) ────
    timeshift_result = _detect_timeshift_pattern(text_lower)
    if timeshift_result:
        result["mode"] = "timeshift"
        result["dataset"] = timeshift_result["dataset"]
        result["buy_hours"] = timeshift_result["buy_hours"]
        result["sell_hours"] = timeshift_result["sell_hours"]
        # Extract areas if mentioned
        detected_areas = set()
        for country, code in COUNTRY_MAP.items():
            if re.search(r"\b" + re.escape(country) + r"\b", text_lower):
                detected_areas.add(code)
        for a in AREAS:
            if re.search(r"\barea\s+" + re.escape(a.lower()) + r"\b", text_lower):
                detected_areas.add(a)
            if a not in ("AT", "BE"):
                if re.search(r"(?:^|[\s,])" + re.escape(a.lower()) + r"(?:$|[\s,])", text_lower):
                    detected_areas.add(a)
        result["areas"] = sorted(detected_areas)
        # Parse date filters
        date_match = re.search(
            r"(?:last|past|recent)\s+(?:one\s+|1\s+)?(\d*)\s*(day|week|month|year)s?",
            text_lower)
        if date_match:
            num = int(date_match.group(1)) if date_match.group(1) else 1
            unit = date_match.group(2)
            if unit == "day":
                result["date_filter_days"] = num
            elif unit == "week":
                result["date_filter_days"] = num * 7
            elif unit == "month":
                result["date_filter_days"] = num * 30
            elif unit == "year":
                result["date_filter_days"] = num * 365
        return result

    multi_leg_result = _detect_multi_leg_pattern(text_lower)
    if multi_leg_result:
        result["mode"] = "multi_leg"
        result["legs"] = multi_leg_result
        # Find datasets
        datasets_found = _find_datasets_in_text(text_lower)
        if len(datasets_found) >= 2:
            result["buy_dataset"] = datasets_found[0]
            result["sell_dataset"] = datasets_found[1]
        elif len(datasets_found) == 1:
            # Use same dataset for both, or might be missing sell_dataset
            result["buy_dataset"] = datasets_found[0]
            result["missing"].append("sell_dataset")
        else:
            result["missing"].append("buy_dataset")
            result["missing"].append("sell_dataset")
        # Parse date filters
        date_match = re.search(
            r"(?:last|past|recent)\s+(?:one\s+|1\s+)?(\d*)\s*(day|week|month|year)s?",
            text_lower)
        if date_match:
            num = int(date_match.group(1)) if date_match.group(1) else 1
            unit = date_match.group(2)
            if unit == "day":
                result["date_filter_days"] = num
            elif unit == "week":
                result["date_filter_days"] = num * 7
            elif unit == "month":
                result["date_filter_days"] = num * 30
            elif unit == "year":
                result["date_filter_days"] = num * 365
        return result

    # ── Detect "all areas" intent ─────────────────────────────────────
    all_areas_intent = bool(re.search(
        r"\b(?:all\s+areas?|every\s+area|each\s+area|all\s+countries)\b", text_lower
    ))

    # ── Detect lookup mode ────────────────────────────────────────────
    # Broader set of trigger words for data viewing
    lookup_triggers = bool(re.search(
        r"\b(?:show|pull|get|display|fetch|view|table|list|prices?"
        r"|full|everything|all\s+data|give\s+me|dump|export|raw"
        r"|overview|summary|what\s+is|what\s+are|how\s+much)\b",
        text_lower
    ))
    has_buy_sell = bool(re.search(r"\b(?:buy|sell|buying|selling)\b", text_lower))
    has_compare = bool(re.search(r"\b(?:compare|vs\.?|versus|against|spread)\b", text_lower))

    # Find all datasets mentioned (using safe boundary matching)
    datasets_found = _find_datasets_in_text(text_lower)

    # ── Detect MULTIPLE areas (early, needed for mode detection) ────────
    detected_areas = set()
    for country, code in COUNTRY_MAP.items():
        if re.search(r"\b" + re.escape(country) + r"\b", text_lower):
            detected_areas.add(code)
    for a in AREAS:
        if re.search(r"\barea\s+" + re.escape(a.lower()) + r"\b", text_lower):
            detected_areas.add(a)
    for a in AREAS:
        if a in ("AT", "BE"):
            continue
        if re.search(r"(?:^|[\s,])" + re.escape(a.lower()) + r"(?:$|[\s,])", text_lower):
            detected_areas.add(a)

    # ── Detect cross-dataset cross-area pairs (e.g. "DA France vs IDA1 Germany")
    is_cross = False
    cross_pairs = []
    if len(datasets_found) >= 2 and len(detected_areas) >= 2:
        cross_pairs = _parse_dataset_area_pairs(text_lower, datasets_found)
        # If each dataset got a DIFFERENT area, this is a cross-area trade
        paired_areas = [a for _, a in cross_pairs if a is not None]
        if len(paired_areas) >= 2 and len(set(paired_areas)) >= 2:
            is_cross = True

    # ── Detect same-dataset cross-area trades ────────────────────────
    # "buy dayahead Germany sell dayahead Belgium" → cross mode with same dataset
    if not is_cross and has_buy_sell and len(detected_areas) >= 2:
        # Try to pair each area with buy/sell by proximity in the text
        buy_pos = re.search(r"\bbuy(?:ing)?\b", text_lower)
        sell_pos = re.search(r"\bsell(?:ing)?\b", text_lower)
        if buy_pos and sell_pos:
            buy_idx = buy_pos.start()
            sell_idx = sell_pos.start()
            # Find which area is closest to "buy" and which to "sell"
            area_positions = {}
            for a in detected_areas:
                # Find position of area code or country name
                code_lower = a.lower()
                for country, code in COUNTRY_MAP.items():
                    if code == a:
                        m_pos = re.search(r"\b" + re.escape(country) + r"\b", text_lower)
                        if m_pos:
                            area_positions[a] = m_pos.start()
                            break
                if a not in area_positions:
                    m_pos = re.search(r"\b" + re.escape(code_lower) + r"\b", text_lower)
                    if m_pos:
                        area_positions[a] = m_pos.start()

            if len(area_positions) >= 2:
                # Assign areas: area closest after "buy" → buy_area, closest after "sell" → sell_area
                buy_area_cand = None
                sell_area_cand = None
                for a, pos in area_positions.items():
                    if pos > buy_idx and pos < sell_idx:
                        buy_area_cand = a
                    elif pos > sell_idx:
                        if sell_area_cand is None or pos < area_positions.get(sell_area_cand, 9999):
                            sell_area_cand = a

                if buy_area_cand and sell_area_cand and buy_area_cand != sell_area_cand:
                    is_cross = True
                    ds = datasets_found[0] if datasets_found else "dayahead"
                    cross_pairs = [(ds, buy_area_cand), (ds, sell_area_cand)]

    # Determine mode
    if is_cross:
        # "DA France vs IDA1 Germany" → cross-dataset cross-area strategy
        is_lookup = False
        is_compare = False
    elif has_buy_sell:
        # Explicit buy/sell → strategy mode
        is_lookup = False
        is_compare = False
    elif has_compare and len(datasets_found) >= 2:
        # "compare DA and IDA1" / "DA vs IDA1" → strategy (buy first, sell second)
        is_lookup = False
        is_compare = False
    elif has_compare and len(datasets_found) == 1 and len(detected_areas) >= 2:
        # "DA Germany vs France" → compare same dataset across areas
        is_lookup = False
        is_compare = True
    elif has_compare and len(datasets_found) <= 1 and len(detected_areas) >= 2:
        # "compare Germany and France" → compare default dataset across areas
        is_lookup = False
        is_compare = True
    elif len(datasets_found) == 1 and len(detected_areas) >= 2 and not has_buy_sell:
        # "DA GER FR last month" — single dataset, multiple areas, no buy/sell
        # → compare mode (show both areas on same dataset)
        is_lookup = False
        is_compare = True
    elif lookup_triggers and not has_compare:
        # Only treat as lookup if there's a dataset mention OR a strong data keyword
        # Avoid catching "What can you do?" or "hello" as lookup
        has_strong_data_intent = bool(re.search(
            r"\b(?:show|pull|get|display|fetch|view|table|list|dump|export|raw"
            r"|full|everything|all\s+data|give\s+me|overview|summary)\b",
            text_lower
        ))
        if datasets_found or detected_areas or has_strong_data_intent:
            is_lookup = True
            is_compare = False
        else:
            # Weak match like "what is..." with no dataset/area → unrecognized
            is_lookup = False
            is_compare = False
            result["mode"] = "unrecognized"
    elif len(datasets_found) == 1 and not has_buy_sell:
        # Single dataset mentioned, no buy/sell → treat as lookup
        is_lookup = True
        is_compare = False
    elif len(datasets_found) == 0 and not has_buy_sell:
        # No datasets, no buy/sell → unrecognized (can't infer intent)
        is_lookup = False
        is_compare = False
        result["mode"] = "unrecognized"
    else:
        is_lookup = False
        is_compare = False

    if is_cross:
        result["mode"] = "cross"
        result["buy_dataset"] = cross_pairs[0][0]
        result["buy_area"] = cross_pairs[0][1]
        result["sell_dataset"] = cross_pairs[1][0]
        result["sell_area"] = cross_pairs[1][1]
        result["areas"] = sorted(detected_areas)
    elif is_compare:
        result["mode"] = "compare"
        result["compare_dataset"] = datasets_found[0] if datasets_found else "dayahead"
        result["areas"] = sorted(detected_areas)
        if len(result["areas"]) < 2:
            result["missing"].append("areas")  # Compare needs at least 2 areas
    elif is_lookup:
        result["mode"] = "lookup"
        if datasets_found:
            result["lookup_dataset"] = datasets_found[0]
            # Support multi-dataset lookups (e.g. "show IDA1 IDA2 IDA3 for Belgium")
            if len(datasets_found) >= 2 and not has_buy_sell and not has_compare:
                # De-duplicate by file (e.g. "dayahead" and "da" → same file)
                unique_ds = []
                seen_files = set()
                for ds in datasets_found:
                    f = DATASET_MAP[ds][0]
                    if f not in seen_files:
                        unique_ds.append(ds)
                        seen_files.add(f)
                if len(unique_ds) >= 2:
                    result["lookup_datasets"] = unique_ds
        else:
            result["lookup_dataset"] = "dayahead"

    # ── Buy / sell datasets (strategy mode) ─────────────────────────────
    if not is_lookup and not is_cross and not is_compare:
        # Try explicit buy/sell patterns
        buy_match = re.search(
            r"buy(?:ing)?\s+(?:on\s+|from\s+)?([a-z0-9\- ]+?)(?:,|\band\b|\bsell(?:ing)?\b|\bhours?\b|\bweek|\barea\b|\bfor\b|$)",
            text_lower)
        sell_match = re.search(
            r"sell(?:ing)?\s+(?:on\s+)?([a-z0-9\- ]+?)(?:,|\band\b|\bbuy(?:ing)?\b|\bhours?\b|\bweek|\barea\b|\bfor\b|$)",
            text_lower)

        if buy_match:
            key = buy_match.group(1).strip()
            if key in DATASET_MAP:
                result["buy_dataset"] = key
        if sell_match:
            key = sell_match.group(1).strip()
            if key in DATASET_MAP:
                result["sell_dataset"] = key

        # Fallback: use datasets_found order (first=buy, second=sell)
        if result["buy_dataset"] is None and len(datasets_found) >= 1:
            result["buy_dataset"] = datasets_found[0]
        if result["sell_dataset"] is None and len(datasets_found) >= 2:
            # Avoid picking the same dataset as buy
            for ds in datasets_found[1:]:
                if DATASET_MAP[ds][0] != DATASET_MAP.get(result["buy_dataset"], ("",))[0]:
                    result["sell_dataset"] = ds
                    break

        if not result["buy_dataset"]:
            result["missing"].append("buy_dataset")
        if not result["sell_dataset"]:
            result["missing"].append("sell_dataset")

    # Areas already detected above (before mode detection).
    # For non-compare modes, set them now. Compare mode set them already.
    if result["mode"] != "compare":
        result["areas"] = sorted(detected_areas)
    # Areas are NEVER required for strategy/lookup — empty means "all areas".

    # ── Detect MULTIPLE hour ranges ─────────────────────────────────────
    hour_matches = re.finditer(
        r"(?:hours?|between|from)\s*(\d{1,2})\s*(?:to|-|and)\s*(\d{1,2})",
        text_lower)
    for m in hour_matches:
        h1, h2 = int(m.group(1)), int(m.group(2))
        result["hour_ranges"].append((min(h1, h2), max(h1, h2)))

    # Also catch "X to Y hours" pattern
    hour_matches2 = re.finditer(
        r"(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*hours?",
        text_lower)
    for m in hour_matches2:
        h1, h2 = int(m.group(1)), int(m.group(2))
        pair = (min(h1, h2), max(h1, h2))
        if pair not in result["hour_ranges"]:
            result["hour_ranges"].append(pair)

    # ── Detect days ─────────────────────────────────────────────────────
    days_set = set()
    for word, val in DAY_MAP.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            if val == "weekdays":
                days_set.update([0, 1, 2, 3, 4])
            elif val == "weekends":
                days_set.update([5, 6])
            else:
                days_set.add(val)
    if days_set:
        result["days"] = sorted(days_set)

    # ── Detect date range filter ────────────────────────────────────────
    # "last one month", "last 2 weeks", "last 30 days", "past month"
    date_match = re.search(
        r"(?:last|past|recent)\s+(?:one\s+|1\s+)?(\d*)\s*(day|week|month|year)s?",
        text_lower)
    if not date_match:
        # Fallback: "for 30 days", "30 days", "for us 30 days" (common typo)
        date_match = re.search(
            r"(?:for\s+(?:\w+\s+)?)?(\d+)\s*(day|week|month|year)s?\b",
            text_lower)
    if date_match:
        num = int(date_match.group(1)) if date_match.group(1) else 1
        unit = date_match.group(2)
        if unit == "day":
            result["date_filter_days"] = num
        elif unit == "week":
            result["date_filter_days"] = num * 7
        elif unit == "month":
            result["date_filter_days"] = num * 30
        elif unit == "year":
            result["date_filter_days"] = num * 365

    return result


# ── Lookup executor & renderer ─────────────────────────────────────────────

def execute_lookup(params: Dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Load a single dataset and apply filters. Returns (df, error_msg)."""
    key = params["lookup_dataset"]
    filename, price_col = DATASET_MAP[key]
    df = load_dataset(filename, price_col)
    if df is None:
        return None, f"Data file `{filename}` not found."

    # Filter by areas
    if params.get("areas"):
        df = df[df["area"].isin(params["areas"])]

    # Filter by days
    if params.get("days") is not None:
        df = df[df["day_of_week"].isin(params["days"])]

    # Filter by date range
    if params.get("date_filter_days"):
        max_date = df["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=params["date_filter_days"])
        df = df[df["deliveryStartCET"] >= cutoff]

    # Filter by hour ranges
    if params.get("hour_ranges"):
        masks = [((df["hour"] >= h[0]) & (df["hour"] <= h[1])) for h in params["hour_ranges"]]
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        df = df[combined_mask]

    if df.empty:
        return None, "No data matches the given filters."

    return df.sort_values("deliveryStartCET"), ""


def render_lookup(df: pd.DataFrame, params: Dict):
    """Render lookup results: KPIs, price chart, and data table."""
    filename = DATASET_MAP[params["lookup_dataset"]][0]
    ds_name = DATASET_DISPLAY.get(filename, params["lookup_dataset"].upper())

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dataset", ds_name)
    k2.metric("Avg Price", f"\u20ac{df['price_value'].mean():,.2f}")
    k3.metric("Min / Max", f"\u20ac{df['price_value'].min():,.1f} / \u20ac{df['price_value'].max():,.1f}")
    k4.metric("Records", f"{len(df):,}")

    areas_in_data = sorted(df["area"].unique())
    area_colors = PALETTE["area"]

    # Price chart + daily avg side by side
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        for a in areas_in_data:
            adf = df[df["area"] == a]
            fig.add_trace(go.Scatter(
                x=adf["deliveryStartCET"], y=adf["price_value"],
                mode="lines", name=a,
                line=dict(color=area_colors.get(a, "#666"), width=1.5),
                opacity=0.7,
            ))
        fig.update_layout(title=f"{ds_name} Prices", xaxis_title="", yaxis_title="EUR/MWh")
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        daily = (df.groupby(["date_cet", "area"])
                 .agg(avg=("price_value", "mean"))
                 .reset_index().sort_values("date_cet"))
        fig2 = go.Figure()
        for a in areas_in_data:
            adf = daily[daily["area"] == a]
            fig2.add_trace(go.Bar(
                x=adf["date_cet"], y=adf["avg"], name=a,
                marker_color=area_colors.get(a, "#666"), opacity=0.85,
            ))
        fig2.update_layout(title="Daily Avg Price", xaxis_title="", yaxis_title="EUR/MWh",
                           barmode="group")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Data table
    show_cols = ["date_cet", "area", "deliveryStartCET", "hour", "price_value"]
    st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=350)


def render_lookup_multi(df: pd.DataFrame, params: Dict):
    """Render multi-dataset lookup results: one section per dataset."""
    ds_list = params.get("lookup_datasets", [params.get("lookup_dataset", "dayahead")])
    ds_colors = {"IDA1": "#1f77b4", "IDA2": "#ff7f0e", "IDA3": "#2ca02c",
                 "DayAhead": "#d62728", "VWAP": "#9467bd"}

    # Summary KPIs row
    st.markdown(f"### Comparing {len(ds_list)} Datasets")
    cols = st.columns(len(ds_list))
    for i, ds_key in enumerate(ds_list):
        lf = DATASET_MAP[ds_key][0]
        ds_label = DATASET_DISPLAY.get(lf, ds_key.upper())
        sub = df[df["_dataset"] == ds_label]
        with cols[i]:
            st.metric(ds_label, f"\u20ac{sub['price_value'].mean():,.2f} avg")
            st.caption(f"{len(sub):,} records | Min \u20ac{sub['price_value'].min():,.1f} / Max \u20ac{sub['price_value'].max():,.1f}")

    areas_in_data = sorted(df["area"].unique())

    # Combined overlay chart
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        for ds_key in ds_list:
            lf = DATASET_MAP[ds_key][0]
            ds_label = DATASET_DISPLAY.get(lf, ds_key.upper())
            sub = df[df["_dataset"] == ds_label]
            for a in areas_in_data:
                asub = sub[sub["area"] == a]
                if not asub.empty:
                    name = f"{ds_label} ({a})" if len(areas_in_data) > 1 else ds_label
                    fig.add_trace(go.Scatter(
                        x=asub["deliveryStartCET"], y=asub["price_value"],
                        mode="lines", name=name,
                        line=dict(color=ds_colors.get(ds_label, "#666"), width=1.5),
                        opacity=0.7,
                    ))
        fig.update_layout(title="Price Comparison", xaxis_title="", yaxis_title="EUR/MWh")
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        daily = (df.groupby(["date_cet", "_dataset"])
                 .agg(avg=("price_value", "mean"))
                 .reset_index().sort_values("date_cet"))
        fig2 = go.Figure()
        for ds_key in ds_list:
            lf = DATASET_MAP[ds_key][0]
            ds_label = DATASET_DISPLAY.get(lf, ds_key.upper())
            dsub = daily[daily["_dataset"] == ds_label]
            fig2.add_trace(go.Bar(
                x=dsub["date_cet"], y=dsub["avg"], name=ds_label,
                marker_color=ds_colors.get(ds_label, "#666"), opacity=0.85,
            ))
        fig2.update_layout(title="Daily Avg Price by Dataset", xaxis_title="", yaxis_title="EUR/MWh",
                           barmode="group")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Per-dataset sections
    for ds_key in ds_list:
        lf = DATASET_MAP[ds_key][0]
        ds_label = DATASET_DISPLAY.get(lf, ds_key.upper())
        sub = df[df["_dataset"] == ds_label]
        with st.expander(f"{ds_label} — {len(sub):,} records", expanded=False):
            show_cols = ["date_cet", "area", "deliveryStartCET", "hour", "price_value"]
            st.dataframe(sub[show_cols].reset_index(drop=True), use_container_width=True, height=250)


# ── Cross-area comparison executor & renderer ──────────────────────────────

def execute_compare(params: Dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Load a single dataset, pivot by area, compute spread. Returns (df, error)."""
    key = params["compare_dataset"]
    filename, price_col = DATASET_MAP[key]
    df = load_dataset(filename, price_col)
    if df is None:
        return None, f"Data file `{filename}` not found."

    areas = params.get("areas", [])
    if len(areas) < 2:
        return None, "Cross-area comparison needs at least 2 areas."
    df = df[df["area"].isin(areas)]

    # Filter by days
    if params.get("days") is not None:
        df = df[df["day_of_week"].isin(params["days"])]

    # Filter by date range
    if params.get("date_filter_days"):
        max_date = df["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=params["date_filter_days"])
        df = df[df["deliveryStartCET"] >= cutoff]

    # Filter by hour ranges
    if params.get("hour_ranges"):
        masks = [((df["hour"] >= h[0]) & (df["hour"] <= h[1])) for h in params["hour_ranges"]]
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        df = df[combined_mask]

    if df.empty:
        return None, "No data matches the given filters."

    return df.sort_values("deliveryStartCET"), ""


def render_compare(df: pd.DataFrame, params: Dict):
    """Render cross-area comparison: spread chart, KPIs, heatmap, table."""
    key = params["compare_dataset"]
    filename = DATASET_MAP[key][0]
    ds_name = DATASET_DISPLAY.get(filename, key.upper())
    areas = params["areas"]
    area_colors = PALETTE["area"]

    # Pivot to get one column per area
    pivot = df.pivot_table(
        index=["date_cet", "deliveryStartCET", "hour"],
        columns="area",
        values="price_value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None

    # Which areas actually have data
    available_areas = [a for a in areas if a in pivot.columns]
    if len(available_areas) < 2:
        st.error("Not enough overlapping data for comparison.")
        return

    # Compute spread: first area minus second area
    a1, a2 = available_areas[0], available_areas[1]
    pivot["spread"] = pivot[a1] - pivot[a2]
    pivot = pivot.dropna(subset=["spread"])

    # ── KPI tiles ────────────────────────────────────────────────────
    risk = compute_risk_metrics(pivot["spread"])
    avg_a1 = pivot[a1].mean()
    avg_a2 = pivot[a2].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"Avg {a1}", f"\u20ac{avg_a1:,.2f}")
    k2.metric(f"Avg {a2}", f"\u20ac{avg_a2:,.2f}")
    k3.metric("Avg Spread", f"\u20ac{risk['avg_delta']:+,.2f}")
    k4.metric("Sharpe", f"{risk['sharpe']:+.3f}")
    k5.metric("Total Spread P&L", f"\u20ac{risk['total_pnl']:,.2f}")
    k6.metric("Slots", f"{risk['n_slots']:,}")

    # ── Charts ───────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        # Price overlay
        fig = go.Figure()
        for a in available_areas:
            fig.add_trace(go.Scatter(
                x=pivot["deliveryStartCET"], y=pivot[a],
                mode="lines", name=a,
                line=dict(color=area_colors.get(a, "#666"), width=1.5),
                opacity=0.8,
            ))
        fig.update_layout(title=f"{ds_name} — {a1} vs {a2}", yaxis_title="EUR/MWh")
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Spread chart with cumulative spread
        cum_spread_pnl = (pivot["spread"] * 0.25).cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pivot["deliveryStartCET"], y=cum_spread_pnl,
            mode="lines", name="Cumulative Spread P&L",
            line=dict(color=PALETTE["primary"][0], width=2),
            fill="tozeroy",
            fillcolor="rgba(13,71,161,0.1)",
        ))
        fig2.update_layout(title=f"Cumulative Spread ({a1} - {a2})", yaxis_title="EUR")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Hourly heatmap: avg spread by hour ───────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        daily = pivot.copy()
        daily["date"] = pivot["deliveryStartCET"].dt.date
        hourly_spread = daily.groupby("hour")["spread"].agg(["mean", "std", "count"]).reset_index()
        hourly_spread["sharpe"] = hourly_spread["mean"] / hourly_spread["std"]
        hourly_spread["sharpe"] = hourly_spread["sharpe"].fillna(0)

        fig3 = go.Figure()
        colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0]
                  for v in hourly_spread["mean"]]
        fig3.add_trace(go.Bar(
            x=hourly_spread["hour"], y=hourly_spread["mean"],
            marker_color=colors, name="Avg Spread",
        ))
        fig3.update_layout(
            title=f"Avg Spread by Hour ({a1} - {a2})",
            xaxis_title="Hour", yaxis_title="EUR/MWh",
        )
        _style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Daily average comparison bars
        daily_avg = df.groupby(["date_cet", "area"]).agg(
            avg_price=("price_value", "mean")
        ).reset_index()
        fig4 = go.Figure()
        for a in available_areas:
            adf = daily_avg[daily_avg["area"] == a]
            fig4.add_trace(go.Bar(
                x=adf["date_cet"], y=adf["avg_price"], name=a,
                marker_color=area_colors.get(a, "#666"), opacity=0.85,
            ))
        fig4.update_layout(
            title="Daily Avg Price by Area",
            xaxis_title="", yaxis_title="EUR/MWh", barmode="group",
        )
        _style_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # If more than 2 areas, show all pairwise spreads
    if len(available_areas) > 2:
        st.subheader("All Pairwise Spreads")
        pairs = []
        for i in range(len(available_areas)):
            for j in range(i + 1, len(available_areas)):
                ai, aj = available_areas[i], available_areas[j]
                sp = pivot[ai] - pivot[aj]
                sp = sp.dropna()
                if len(sp) > 0:
                    pairs.append({
                        "Pair": f"{ai} - {aj}",
                        "Avg Spread": round(sp.mean(), 2),
                        "Std": round(sp.std(), 2),
                        "Sharpe": round(sp.mean() / sp.std(), 3) if sp.std() > 0 else 0,
                        "Min": round(sp.min(), 2),
                        "Max": round(sp.max(), 2),
                        "Slots": len(sp),
                    })
        if pairs:
            st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

    # Data table with spread
    show_df = pivot[["date_cet", "deliveryStartCET", "hour"] + available_areas + ["spread"]].copy()
    show_df = show_df.rename(columns={"spread": f"{a1} - {a2}"})
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=350)

    # CSV export
    csv = show_df.to_csv(index=False)
    st.download_button("Download CSV", csv, f"compare_{a1}_{a2}.csv", "text/csv")


# ── Cross-dataset cross-area executor & renderer ───────────────────────────

def execute_cross(params: Dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Execute a cross-dataset cross-area strategy.

    E.g. Buy DA in France, Sell IDA1 in Germany — merges on timestamp only
    (not area), since each side is filtered to its own area.
    """
    buy_key = params["buy_dataset"]
    sell_key = params["sell_dataset"]
    buy_area = params["buy_area"]
    sell_area = params["sell_area"]

    buy_file, buy_col = DATASET_MAP[buy_key]
    sell_file, sell_col = DATASET_MAP[sell_key]

    buy_df = load_dataset(buy_file, buy_col)
    sell_df = load_dataset(sell_file, sell_col)
    if buy_df is None:
        return None, f"Data file `{buy_file}` not found."
    if sell_df is None:
        return None, f"Data file `{sell_file}` not found."

    # Filter each side to its own area
    buy_df = buy_df[buy_df["area"] == buy_area].copy()
    sell_df = sell_df[sell_df["area"] == sell_area].copy()

    if buy_df.empty:
        return None, f"No data for {buy_area} in {DATASET_DISPLAY.get(buy_file, buy_key)}."
    if sell_df.empty:
        return None, f"No data for {sell_area} in {DATASET_DISPLAY.get(sell_file, sell_key)}."

    # Merge on timestamp only (not area)
    buy_cols = ["date_cet", "deliveryStartCET", "hour", "day_of_week", "price_value"]
    sell_cols = ["date_cet", "deliveryStartCET", "price_value"]

    merged = pd.merge(
        buy_df[buy_cols].rename(columns={"price_value": "buy_price"}),
        sell_df[sell_cols].rename(columns={"price_value": "sell_price"}),
        on=["date_cet", "deliveryStartCET"],
        how="inner",
        suffixes=("_buy", "_sell"),
    )

    # Filter by days
    if params.get("days") is not None:
        merged = merged[merged["day_of_week"].isin(params["days"])]

    # Filter by date range
    if params.get("date_filter_days"):
        max_date = merged["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=params["date_filter_days"])
        merged = merged[merged["deliveryStartCET"] >= cutoff]

    # Filter by hour ranges
    if params.get("hour_ranges"):
        masks = [((merged["hour"] >= h[0]) & (merged["hour"] <= h[1]))
                 for h in params["hour_ranges"]]
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        merged = merged[combined_mask]

    if merged.empty:
        return None, "No matching data for the given filters."

    merged["delta"] = merged["sell_price"] - merged["buy_price"]
    return merged.sort_values("deliveryStartCET"), ""


def render_cross(df: pd.DataFrame, params: Dict):
    """Render cross-dataset cross-area strategy results."""
    buy_key = params["buy_dataset"]
    sell_key = params["sell_dataset"]
    buy_area = params["buy_area"]
    sell_area = params["sell_area"]

    buy_file = DATASET_MAP[buy_key][0]
    sell_file = DATASET_MAP[sell_key][0]
    buy_label = f"{DATASET_DISPLAY.get(buy_file, buy_key)} ({buy_area})"
    sell_label = f"{DATASET_DISPLAY.get(sell_file, sell_key)} ({sell_area})"

    risk = compute_risk_metrics(df["delta"])
    area_colors = PALETTE["area"]

    # ── KPI tiles ────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total P&L", f"\u20ac{risk['total_pnl']:,.2f}")
    k2.metric("Sharpe Ratio", f"{risk['sharpe']:+.3f}")
    k3.metric("Max Drawdown", f"\u20ac{risk['max_drawdown']:,.2f}")
    k4.metric("Profit Factor", f"{risk['profit_factor']:.2f}")
    k5.metric("Win Rate", f"{risk['win_rate']:.1f}%")
    k6.metric("VaR (95%)", f"\u20ac{risk['var_95']:,.2f}/MWh")

    k7, k8, k9, k10 = st.columns(4)
    k7.metric("Buy", buy_label)
    k8.metric("Avg Buy Price", f"\u20ac{df['buy_price'].mean():,.2f}")
    k9.metric("Sell", sell_label)
    k10.metric("Avg Sell Price", f"\u20ac{df['sell_price'].mean():,.2f}")

    # ── Charts ───────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        # Cumulative P&L with drawdown shading
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=risk["peak"],
            mode="lines", name="Peak",
            line=dict(color="#E0E0E0", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=risk["cum_pnl"],
            mode="lines", name="Cumulative P&L",
            line=dict(color=PALETTE["primary"][0], width=2),
            fill="tonexty", fillcolor="rgba(229,57,53,0.1)",
        ))
        fig.update_layout(
            title=f"Cumulative P&L: Buy {buy_label} → Sell {sell_label}",
            yaxis_title="EUR",
        )
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Price overlay
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=df["buy_price"],
            mode="lines", name=buy_label,
            line=dict(color=area_colors.get(buy_area, PALETTE["primary"][0]), width=1.5),
            opacity=0.8,
        ))
        fig2.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=df["sell_price"],
            mode="lines", name=sell_label,
            line=dict(color=area_colors.get(sell_area, PALETTE["accent"][0]), width=1.5),
            opacity=0.8,
        ))
        fig2.update_layout(title="Price Comparison", yaxis_title="EUR/MWh")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Spread by hour + delta scatter ──────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        hourly = df.groupby("hour")["delta"].agg(["mean", "std", "count"]).reset_index()
        colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0]
                  for v in hourly["mean"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=hourly["hour"], y=hourly["mean"],
                              marker_color=colors, name="Avg Delta"))
        fig3.update_layout(title="Avg Spread by Hour", xaxis_title="Hour",
                           yaxis_title="EUR/MWh")
        _style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=df["delta"],
            mode="markers", name="Delta",
            marker=dict(size=3, color=PALETTE["primary"][2], opacity=0.5),
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="#9E9E9E")
        fig4.update_layout(title="Spread Delta Scatter", yaxis_title="EUR/MWh")
        _style_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Data table + CSV ────────────────────────────────────────────
    show_cols = ["date_cet", "deliveryStartCET", "hour", "buy_price", "sell_price", "delta"]
    st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=350)

    csv = df[show_cols].to_csv(index=False)
    st.download_button("Download CSV", csv,
                       f"cross_{buy_key}_{buy_area}_{sell_key}_{sell_area}.csv", "text/csv")


# ── Strategy executor ───────────────────────────────────────────────────────

def execute_multi_strategy(strategy: Dict) -> Tuple[Optional[List[Tuple[pd.DataFrame, str]]], str]:
    """Execute strategy for each combination of areas x hour_ranges.

    Returns a list of (DataFrame, label) tuples, or (None, error_msg).
    """
    buy_key = strategy["buy_dataset"]
    sell_key = strategy["sell_dataset"]
    buy_file, buy_col = DATASET_MAP[buy_key]
    sell_file, sell_col = DATASET_MAP[sell_key]

    buy_df = load_dataset(buy_file, buy_col)
    sell_df = load_dataset(sell_file, sell_col)

    if buy_df is None:
        return None, f"Data file `{buy_file}` not found."
    if sell_df is None:
        return None, f"Data file `{sell_file}` not found."

    # Prepare columns for merge — include volume if available
    buy_cols = ["date_cet", "area", "deliveryStartCET", "hour", "day_of_week", "price_value"]
    sell_cols = ["date_cet", "area", "deliveryStartCET", "price_value"]
    buy_rename = {"price_value": "buy_price"}
    sell_rename = {"price_value": "sell_price"}

    if "volume" in buy_df.columns:
        buy_cols.append("volume")
        buy_rename["volume"] = "buy_volume"
    if "volume" in sell_df.columns:
        sell_cols.append("volume")
        sell_rename["volume"] = "sell_volume"

    merged = pd.merge(
        buy_df[buy_cols].rename(columns=buy_rename),
        sell_df[sell_cols].rename(columns=sell_rename),
        on=["date_cet", "area", "deliveryStartCET"], how="inner",
    )

    # Compute combined volume (use sell-side volume if available, else buy-side)
    if "sell_volume" in merged.columns:
        merged["volume"] = merged["sell_volume"]
    elif "buy_volume" in merged.columns:
        merged["volume"] = merged["buy_volume"]

    # Filter by areas
    areas = strategy.get("areas", [])
    if areas:
        merged = merged[merged["area"].isin(areas)]

    # Filter by days
    if strategy.get("days") is not None:
        merged = merged[merged["day_of_week"].isin(strategy["days"])]

    # Filter by date range
    if strategy.get("date_filter_days"):
        max_date = merged["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=strategy["date_filter_days"])
        merged = merged[merged["deliveryStartCET"] >= cutoff]

    if merged.empty:
        return None, "No matching data for the given filters. Try broader criteria."

    merged["delta"] = merged["sell_price"] - merged["buy_price"]

    hour_ranges = strategy.get("hour_ranges", [])
    buy_display = DATASET_DISPLAY.get(buy_file, buy_key.upper())
    sell_display = DATASET_DISPLAY.get(sell_file, sell_key.upper())

    # If no hour ranges specified, return one strategy with all hours
    if not hour_ranges:
        hour_ranges = [None]

    results = []
    for hr in hour_ranges:
        subset = merged.copy()
        if hr is not None:
            subset = subset[(subset["hour"] >= hr[0]) & (subset["hour"] <= hr[1])]

        if subset.empty:
            continue

        label = f"Buy {buy_display} -> Sell {sell_display}"
        if hr:
            label += f" | {hr[0]}:00-{hr[1]}:00"
        else:
            label += " | All hours"
        results.append((subset, label))

    if not results:
        return None, "No matching data after applying hour filters."

    return results, ""


# ── Charts ──────────────────────────────────────────────────────────────────

# Professional color palette
PALETTE = {
    "primary":   ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"],
    "accent":    ["#E65100", "#EF6C00", "#F57C00", "#FB8C00", "#FFA726"],
    "green":     ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"],
    "area": {
        "FR": "#0D47A1", "GER": "#E65100", "NL": "#1B5E20",
        "BE": "#6A1B9A", "AT": "#00838F",
    },
}
CHART_HEIGHT = 320
CHART_FONT = dict(family="Inter, system-ui, sans-serif", size=12)
CHART_MARGIN = dict(l=50, r=20, t=40, b=40)

def _style_fig(fig, height=CHART_HEIGHT):
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=CHART_MARGIN,
        font=CHART_FONT,
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        xaxis=dict(gridcolor="#E0E0E0", zeroline=False),
        yaxis=dict(gridcolor="#E0E0E0", zerolinecolor="#9E9E9E", zerolinewidth=1),
    )
    return fig


def compute_risk_metrics(deltas: pd.Series) -> Dict:
    """Compute risk-adjusted metrics for a series of spread deltas."""
    pnl = deltas * 0.25  # EUR per slot (1 MW * 0.25h)
    cum_pnl = pnl.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak

    total_pnl = pnl.sum()
    avg_delta = deltas.mean()
    std_delta = deltas.std()
    sharpe = avg_delta / std_delta if std_delta > 0 else 0
    max_dd = drawdown.min()
    calmar = total_pnl / abs(max_dd) if max_dd != 0 else 0

    gross_wins = pnl[pnl > 0].sum()
    gross_losses = abs(pnl[pnl < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    var_95 = deltas.quantile(0.05)  # 5th percentile = 95% VaR
    win_rate = (deltas > 0).mean() * 100

    return {
        "total_pnl": total_pnl,
        "avg_delta": avg_delta,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "var_95": var_95,
        "win_rate": win_rate,
        "n_slots": len(deltas),
        "cum_pnl": cum_pnl,
        "peak": peak,
        "drawdown": drawdown,
    }


def execute_timeshift(params: Dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Execute time-shift strategy: buy and sell on SAME dataset at DIFFERENT hours.

    Returns (df, error_msg) with columns: date, area, buy_avg, sell_avg, delta, pnl
    """
    dataset_key = params.get("dataset")
    if not dataset_key:
        return None, "No dataset specified for timeshift mode."

    buy_hours = params.get("buy_hours")
    sell_hours = params.get("sell_hours")
    if not buy_hours or not sell_hours:
        return None, "Buy and sell hour windows required for timeshift mode."

    filename, price_col = DATASET_MAP.get(dataset_key, (None, None))
    if filename is None:
        return None, f"Unknown dataset: {dataset_key}"

    df = load_dataset(filename, price_col)
    if df is None:
        return None, f"Data file `{filename}` not found."

    areas = params.get("areas", [])
    if areas:
        df = df[df["area"].isin(areas)]

    # Filter by days
    if params.get("days") is not None:
        df = df[df["day_of_week"].isin(params["days"])]

    # Filter by date range
    if params.get("date_filter_days"):
        max_date = df["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=params["date_filter_days"])
        df = df[df["deliveryStartCET"] >= cutoff]

    if df.empty:
        return None, "No data matches the given filters."

    # Compute daily statistics
    df["date"] = df["deliveryStartCET"].dt.date

    results = []
    for (date, area), group in df.groupby(["date", "area"]):
        buy_data = group[(group["hour"] >= buy_hours[0]) & (group["hour"] <= buy_hours[1])]
        sell_data = group[(group["hour"] >= sell_hours[0]) & (group["hour"] <= sell_hours[1])]

        if len(buy_data) > 0 and len(sell_data) > 0:
            buy_avg = buy_data["price_value"].mean()
            sell_avg = sell_data["price_value"].mean()
            delta = sell_avg - buy_avg
            # P&L per day: delta * volume_factor * num_quarter_hours (assuming 1 MW, 0.25h slots)
            num_slots = min(len(buy_data), len(sell_data))
            pnl = delta * 0.25 * num_slots

            results.append({
                "date": date,
                "area": area,
                "buy_avg": buy_avg,
                "sell_avg": sell_avg,
                "delta": delta,
                "pnl": pnl,
            })

    if not results:
        return None, "No valid buy/sell hour windows found in the data."

    result_df = pd.DataFrame(results)
    return result_df, ""


def render_timeshift(df: pd.DataFrame, params: Dict):
    """Render time-shift strategy results."""
    dataset_key = params.get("dataset")
    filename = DATASET_MAP.get(dataset_key, (None,))[0]
    ds_name = DATASET_DISPLAY.get(filename, dataset_key.upper()) if filename else dataset_key.upper()

    buy_hours = params.get("buy_hours", (0, 0))
    sell_hours = params.get("sell_hours", (0, 0))

    # Risk metrics
    risk = compute_risk_metrics(df["delta"])
    area_colors = PALETTE["area"]

    # KPI tiles
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Dataset", ds_name)
    k2.metric("Total P&L", f"€{risk['total_pnl']:,.2f}")
    k3.metric("Avg Delta", f"€{risk['avg_delta']:+,.2f}/MWh")
    k4.metric("Sharpe", f"{risk['sharpe']:+.3f}")
    k5.metric("Win Rate", f"{risk['win_rate']:.1f}%")
    k6.metric("Days", f"{len(df):,}")

    st.caption(
        f"**Buy hours:** {buy_hours[0]}:00-{buy_hours[1]}:00 | "
        f"**Sell hours:** {sell_hours[0]}:00-{sell_hours[1]}:00"
    )

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        # Cumulative P&L
        cum_pnl = df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=cum_pnl,
            mode="lines", name="Cumulative P&L",
            line=dict(color=PALETTE["primary"][0], width=2),
            fill="tozeroy", fillcolor="rgba(13,71,161,0.1)",
        ))
        fig.update_layout(title="Cumulative P&L", xaxis_title="", yaxis_title="EUR")
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Daily P&L bar chart
        colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0] for v in df["pnl"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df["date"], y=df["pnl"],
            marker_color=colors, name="Daily P&L",
        ))
        fig2.update_layout(title="Daily P&L", xaxis_title="", yaxis_title="EUR")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Price profile by area/hour
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Avg buy price vs sell price by area
        daily_avg = df.groupby("area")[["buy_avg", "sell_avg"]].mean().reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=daily_avg["area"], y=daily_avg["buy_avg"],
            name="Buy Avg", marker_color=PALETTE["primary"][0],
        ))
        fig3.add_trace(go.Bar(
            x=daily_avg["area"], y=daily_avg["sell_avg"],
            name="Sell Avg", marker_color=PALETTE["accent"][0],
        ))
        fig3.update_layout(
            title="Avg Buy vs Sell Price by Area",
            xaxis_title="Area", yaxis_title="EUR/MWh", barmode="group"
        )
        _style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Delta by area
        delta_by_area = df.groupby("area")["delta"].mean().reset_index()
        colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0] for v in delta_by_area["delta"]]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=delta_by_area["area"], y=delta_by_area["delta"],
            marker_color=colors, name="Avg Delta",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="#BDBDBD")
        fig4.update_layout(
            title="Avg Spread by Area",
            xaxis_title="Area", yaxis_title="EUR/MWh"
        )
        _style_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # Data table
    show_df = df[["date", "area", "buy_avg", "sell_avg", "delta", "pnl"]].copy()
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=350)

    # CSV export
    csv = show_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "timeshift_results.csv", "text/csv")


def execute_multi_leg(params: Dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Execute multi-leg volume strategy: weighted P&L across multiple volume legs.

    Returns (df, error_msg) with merged data and weighted P&L calculations.
    """
    legs = params.get("legs", [])
    if not legs or len(legs) < 2:
        return None, "Multi-leg strategy requires at least 2 legs."

    buy_key = params.get("buy_dataset")
    sell_key = params.get("sell_dataset")
    if not buy_key or not sell_key:
        return None, "Both buy and sell datasets required for multi-leg strategy."

    buy_file, buy_col = DATASET_MAP.get(buy_key, (None, None))
    sell_file, sell_col = DATASET_MAP.get(sell_key, (None, None))

    if buy_file is None or sell_file is None:
        return None, "Unknown dataset(s) specified."

    buy_df = load_dataset(buy_file, buy_col)
    sell_df = load_dataset(sell_file, sell_col)

    if buy_df is None:
        return None, f"Data file `{buy_file}` not found."
    if sell_df is None:
        return None, f"Data file `{sell_file}` not found."

    # Merge on timestamp and area
    buy_cols = ["date_cet", "area", "deliveryStartCET", "hour", "day_of_week", "price_value"]
    sell_cols = ["date_cet", "area", "deliveryStartCET", "price_value"]

    merged = pd.merge(
        buy_df[buy_cols].rename(columns={"price_value": "buy_price"}),
        sell_df[sell_cols].rename(columns={"price_value": "sell_price"}),
        on=["date_cet", "area", "deliveryStartCET"], how="inner",
    )

    # Filter by days
    if params.get("days") is not None:
        merged = merged[merged["day_of_week"].isin(params["days"])]

    # Filter by date range
    if params.get("date_filter_days"):
        max_date = merged["deliveryStartCET"].max()
        cutoff = max_date - timedelta(days=params["date_filter_days"])
        merged = merged[merged["deliveryStartCET"] >= cutoff]

    if merged.empty:
        return None, "No matching data for the given filters."

    # Compute weighted P&L
    merged["delta"] = merged["sell_price"] - merged["buy_price"]

    # For each leg, add volume weighting
    leg_volumes = {}
    for leg in legs:
        leg_volumes[leg["area"]] = (leg["side"], leg["volume_mw"])

    # Calculate weighted contribution per slot
    def calc_weighted_pnl(row):
        area = row["area"]
        if area not in leg_volumes:
            return 0
        side, volume = leg_volumes[area]
        if side == "buy":
            return -row["delta"] * volume * 0.25
        else:
            return row["delta"] * volume * 0.25

    merged["weighted_pnl"] = merged.apply(calc_weighted_pnl, axis=1)

    return merged, ""


def render_multi_leg(df: pd.DataFrame, params: Dict):
    """Render multi-leg strategy results."""
    buy_key = params.get("buy_dataset")
    sell_key = params.get("sell_dataset")
    legs = params.get("legs", [])

    buy_file = DATASET_MAP.get(buy_key, (None,))[0]
    sell_file = DATASET_MAP.get(sell_key, (None,))[0]
    buy_label = DATASET_DISPLAY.get(buy_file, buy_key.upper()) if buy_file else buy_key.upper()
    sell_label = DATASET_DISPLAY.get(sell_file, sell_key.upper()) if sell_file else sell_key.upper()

    # Risk metrics
    risk = compute_risk_metrics(df["delta"])
    area_colors = PALETTE["area"]

    # KPI tiles
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total P&L", f"€{df['weighted_pnl'].sum():,.2f}")
    k2.metric("Sharpe Ratio", f"{risk['sharpe']:+.3f}")
    k3.metric("Avg Delta", f"€{risk['avg_delta']:+,.2f}/MWh")
    k4.metric("Win Rate", f"{risk['win_rate']:.1f}%")
    k5.metric("Slots", f"{len(df):,}")
    k6.metric("Legs", f"{len(legs)}")

    # Show leg details
    leg_str = " | ".join(f"{leg['side'].upper()} {leg['volume_mw']}MW {leg['area']}" for leg in legs)
    st.caption(f"**Legs:** {leg_str}")

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        # Cumulative weighted P&L
        cum_pnl = df["weighted_pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["deliveryStartCET"], y=cum_pnl,
            mode="lines", name="Cumulative P&L",
            line=dict(color=PALETTE["primary"][0], width=2),
            fill="tozeroy", fillcolor="rgba(13,71,161,0.1)",
        ))
        fig.update_layout(title="Cumulative Weighted P&L", xaxis_title="", yaxis_title="EUR")
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Per-leg contribution breakdown
        contrib = []
        for leg in legs:
            leg_area = leg["area"]
            leg_data = df[df["area"] == leg_area]
            total_contrib = leg_data["weighted_pnl"].sum()
            contrib.append({"Leg": f"{leg['side'].upper()} {leg['volume_mw']}MW {leg_area}", "P&L": total_contrib})

        contrib_df = pd.DataFrame(contrib)
        colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0] for v in contrib_df["P&L"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=contrib_df["Leg"], y=contrib_df["P&L"],
            marker_color=colors, name="Contribution",
        ))
        fig2.update_layout(title="P&L by Leg", xaxis_title="", yaxis_title="EUR")
        _style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Price comparison by area
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Price overlay by area
        fig3 = go.Figure()
        for area in sorted(df["area"].unique()):
            adf = df[df["area"] == area].sort_values("deliveryStartCET")
            fig3.add_trace(go.Scatter(
                x=adf["deliveryStartCET"], y=adf["buy_price"],
                mode="lines", name=f"{area} Buy",
                line=dict(color=area_colors.get(area, "#666"), width=1.5),
                opacity=0.7,
            ))
            fig3.add_trace(go.Scatter(
                x=adf["deliveryStartCET"], y=adf["sell_price"],
                mode="lines", name=f"{area} Sell",
                line=dict(color=area_colors.get(area, "#666"), width=1.5, dash="dot"),
                opacity=0.7,
            ))
        fig3.update_layout(title="Price Comparison by Area", xaxis_title="", yaxis_title="EUR/MWh")
        _style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Delta scatter by area
        fig4 = go.Figure()
        for area in sorted(df["area"].unique()):
            adf = df[df["area"] == area]
            fig4.add_trace(go.Scatter(
                x=adf["deliveryStartCET"], y=adf["delta"],
                mode="markers", name=area,
                marker=dict(size=4, color=area_colors.get(area, "#666"), opacity=0.5),
            ))
        fig4.add_hline(y=0, line_dash="dash", line_color="#BDBDBD")
        fig4.update_layout(title="Spread Delta by Area", xaxis_title="", yaxis_title="EUR/MWh")
        _style_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # Data table
    show_cols = ["date_cet", "area", "deliveryStartCET", "hour", "buy_price", "sell_price", "delta", "weighted_pnl"]
    st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=350)

    # CSV export
    csv = df[show_cols].to_csv(index=False)
    st.download_button("Download CSV", csv, "multi_leg_results.csv", "text/csv")


def render_multi_results(results: List[Tuple[pd.DataFrame, str]], strategy: Dict):
    """Render compact, side-by-side charts with a polished colour scheme."""

    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    areas = strategy.get("areas", [])
    area_str = ", ".join(areas) if areas else "All"
    day_str = ", ".join(day_names.get(d, str(d)) for d in strategy["days"]) if strategy.get("days") else "All"
    date_str = f"Last {strategy['date_filter_days']} days" if strategy.get("date_filter_days") else "All dates"

    # ── KPI metrics row (always first) ──────────────────────────────────
    summary_rows = []
    for df, label in results:
        for area_code in sorted(df["area"].unique()):
            adf = df[df["area"] == area_code]
            risk = compute_risk_metrics(adf["delta"])
            summary_rows.append({
                "Strategy": label, "Area": area_code,
                "Slots": len(adf),
                "Avg Delta": round(risk["avg_delta"], 2),
                "Sharpe": round(risk["sharpe"], 3),
                "Max DD": round(risk["max_drawdown"], 2),
                "Profit Factor": round(risk["profit_factor"], 2),
                "Total P&L": round(risk["total_pnl"], 2),
                "Win Rate": round(risk["win_rate"], 1),
                "VaR 95%": round(risk["var_95"], 2),
            })

    # Compute overall risk metrics across all results
    all_deltas = pd.concat([df["delta"] for df, _ in results], ignore_index=True)
    overall_risk = compute_risk_metrics(all_deltas)

    # Row 1: Primary KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total P&L", f"\u20ac{overall_risk['total_pnl']:,.2f}")
    k2.metric("Sharpe Ratio", f"{overall_risk['sharpe']:+.3f}")
    k3.metric("Max Drawdown", f"\u20ac{overall_risk['max_drawdown']:,.2f}")
    k4.metric("Profit Factor", f"{overall_risk['profit_factor']:.2f}")
    k5.metric("Win Rate", f"{overall_risk['win_rate']:.1f}%")
    k6.metric("VaR (95%)", f"\u20ac{overall_risk['var_95']:,.2f}/MWh")

    # Row 2: Secondary KPIs (check if volume data is available)
    has_volume = any("volume" in df.columns for df, _ in results)
    if has_volume:
        all_vol = pd.concat([df[["delta", "volume"]] for df, _ in results if "volume" in df.columns], ignore_index=True)
        all_vol = all_vol[all_vol["volume"] > 0]
        if len(all_vol) > 0:
            vw_delta = np.average(all_vol["delta"], weights=all_vol["volume"])
            vw_std = np.sqrt(np.average((all_vol["delta"] - vw_delta)**2, weights=all_vol["volume"]))
            vw_sharpe = vw_delta / vw_std if vw_std > 0 else 0
            total_vol = all_vol["volume"].sum()
        else:
            vw_sharpe, total_vol = 0, 0

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Avg Delta", f"\u20ac{overall_risk['avg_delta']:,.2f}/MWh")
        s2.metric("Calmar Ratio", f"{overall_risk['calmar']:+.3f}")
        s3.metric("Slots Traded", f"{overall_risk['n_slots']:,}")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Delta", f"\u20ac{overall_risk['avg_delta']:,.2f}/MWh")
        s2.metric("Calmar Ratio", f"{overall_risk['calmar']:+.3f}")
        s3.metric("Slots Traded", f"{overall_risk['n_slots']:,}")
    s4.metric("Std Dev", f"\u20ac{all_deltas.std():,.2f}/MWh")
    if has_volume and len(all_vol) > 0:
        s5.metric("Vol-Wt Sharpe", f"{vw_sharpe:+.3f}")

    st.caption(
        f"**Areas:** {area_str} &nbsp;&middot;&nbsp; **Days:** {day_str} &nbsp;&middot;&nbsp; "
        f"**Date range:** {date_str} &nbsp;&middot;&nbsp; 1 MW capacity, 0.25 h slots"
    )

    multiple_strategies = len(results) > 1

    # ── Combined Cumulative P&L (hero chart, full width) ────────────────
    all_pnl = []
    for df, label in results:
        pnl = df[["deliveryStartCET", "delta", "area"]].copy().sort_values("deliveryStartCET")
        pnl["pnl_eur"] = pnl["delta"] * 0.25
        pnl["strategy"] = label
        pnl["cum_pnl"] = pnl.groupby("area")["pnl_eur"].cumsum()
        if multiple_strategies:
            pnl["label"] = pnl["area"] + " \u2014 " + pnl["strategy"].str.split("|").str[-1].str.strip()
        else:
            pnl["label"] = pnl["area"]
        all_pnl.append(pnl)

    combined_pnl = pd.concat(all_pnl, ignore_index=True)

    # Assign area-based colours with dash styles for strategy variants
    unique_labels = sorted(combined_pnl["label"].unique())
    area_colors = PALETTE["area"]
    strategy_dashes = ["solid", "dash", "dot", "dashdot"]

    fig_hero = go.Figure()
    for idx, lbl in enumerate(unique_labels):
        lbl_data = combined_pnl[combined_pnl["label"] == lbl].sort_values("deliveryStartCET")
        area_code = lbl.split(" ")[0] if lbl.split(" ")[0] in area_colors else "FR"
        color = area_colors.get(area_code, PALETTE["primary"][idx % 5])
        dash = strategy_dashes[idx // max(len(areas), 1) % len(strategy_dashes)]
        fig_hero.add_trace(go.Scatter(
            x=lbl_data["deliveryStartCET"], y=lbl_data["cum_pnl"],
            mode="lines", name=lbl,
            line=dict(color=color, width=2.5, dash=dash),
        ))

    # Add drawdown shading
    for idx, lbl in enumerate(unique_labels):
        lbl_data = combined_pnl[combined_pnl["label"] == lbl].sort_values("deliveryStartCET")
        pnl_series = lbl_data["cum_pnl"].values
        peak_series = np.maximum.accumulate(pnl_series)
        dd = pnl_series - peak_series
        fig_hero.add_trace(go.Scatter(
            x=lbl_data["deliveryStartCET"], y=peak_series,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_hero.add_trace(go.Scatter(
            x=lbl_data["deliveryStartCET"], y=pnl_series,
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(192, 57, 43, 0.12)",
        ))

    fig_hero.add_hline(y=0, line_dash="dot", line_color="#BDBDBD", line_width=1)
    fig_hero.update_layout(title="Cumulative P&L",
                           xaxis_title="", yaxis_title="EUR")
    _style_fig(fig_hero, height=370)
    st.plotly_chart(fig_hero, use_container_width=True)

    # ── Side-by-side: Delta scatter + Daily avg delta ───────────────────
    for i, (df, label) in enumerate(results):
        if multiple_strategies:
            st.markdown(f"##### {label}")

        col_left, col_right = st.columns(2)

        # Left: Delta scatter (compact)
        with col_left:
            multi_area = df["area"].nunique() > 1
            color_map = {a: PALETTE["area"].get(a, "#666") for a in df["area"].unique()}
            fig_delta = go.Figure()
            for area_code in sorted(df["area"].unique()):
                adf = df[df["area"] == area_code].sort_values("deliveryStartCET")
                fig_delta.add_trace(go.Scatter(
                    x=adf["deliveryStartCET"], y=adf["delta"],
                    mode="markers", name=area_code,
                    marker=dict(color=color_map[area_code], size=4, opacity=0.5),
                ))
            fig_delta.add_hline(y=0, line_dash="dot", line_color="#BDBDBD", line_width=1)
            fig_delta.update_layout(title="Delta per Slot",
                                    xaxis_title="", yaxis_title="EUR/MWh")
            _style_fig(fig_delta)
            st.plotly_chart(fig_delta, use_container_width=True)

        # Right: Daily average delta (bar chart)
        with col_right:
            daily = (df.groupby(["date_cet", "area"])
                     .agg(avg_delta=("delta", "mean"))
                     .reset_index().sort_values("date_cet"))
            fig_daily = go.Figure()
            for area_code in sorted(daily["area"].unique()):
                adf = daily[daily["area"] == area_code]
                colors = [PALETTE["green"][2] if v >= 0 else PALETTE["accent"][0]
                          for v in adf["avg_delta"]]
                fig_daily.add_trace(go.Bar(
                    x=adf["date_cet"], y=adf["avg_delta"],
                    name=area_code,
                    marker_color=colors if daily["area"].nunique() == 1
                        else PALETTE["area"].get(area_code, "#666"),
                    opacity=0.85,
                ))
            fig_daily.update_layout(title="Daily Avg Delta",
                                    xaxis_title="", yaxis_title="EUR/MWh",
                                    barmode="group")
            fig_daily.add_hline(y=0, line_dash="dot", line_color="#BDBDBD", line_width=1)
            _style_fig(fig_daily)
            st.plotly_chart(fig_daily, use_container_width=True)

        # Per-strategy metrics in a compact row
        for area_code in sorted(df["area"].unique()):
            adf = df[df["area"] == area_code]
            risk = compute_risk_metrics(adf["delta"])
            cols = st.columns([1, 1, 1, 1, 1, 1, 1])
            cols[0].markdown(f"**{area_code}**")
            cols[1].metric("P&L", f"\u20ac{risk['total_pnl']:,.2f}", label_visibility="collapsed")
            cols[2].metric("Sharpe", f"{risk['sharpe']:+.3f}", label_visibility="collapsed")
            cols[3].metric("Max DD", f"\u20ac{risk['max_drawdown']:,.2f}", label_visibility="collapsed")
            cols[4].metric("PF", f"{risk['profit_factor']:.2f}", label_visibility="collapsed")
            cols[5].metric("Win", f"{risk['win_rate']:.1f}%", label_visibility="collapsed")
            cols[6].metric("VaR 95", f"\u20ac{risk['var_95']:,.1f}", label_visibility="collapsed")

    # ── Comparison table (collapsible) ──────────────────────────────────
    if multiple_strategies:
        with st.expander("Strategy comparison table"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ── Raw data (collapsible) ──────────────────────────────────────────
    with st.expander("View raw data"):
        for df, label in results:
            st.caption(label)
            st.dataframe(
                df[["date_cet", "area", "deliveryStartCET", "hour", "day_of_week",
                    "buy_price", "sell_price", "delta"]]
                .sort_values("deliveryStartCET").reset_index(drop=True),
                use_container_width=True, height=250)

    # ── CSV Export ───────────────────────────────────────────────────────
    export_rows = []
    for df, label in results:
        edf = df[["date_cet", "area", "deliveryStartCET", "hour", "day_of_week",
                   "buy_price", "sell_price", "delta"]].copy()
        edf["strategy"] = label
        edf["pnl_eur"] = edf["delta"] * 0.25
        export_rows.append(edf)
    if export_rows:
        export_df = pd.concat(export_rows, ignore_index=True).sort_values("deliveryStartCET")
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name="strategy_results.csv",
            mime="text/csv",
        )


# ── Shared form components ─────────────────────────────────────────────────

def _render_common_form_fields(strat: Dict):
    """Render area, hour, day, date fields shared by both modes. Returns dict of selections."""
    missing = strat.get("missing", [])

    # Areas (multi-select)
    area_options = [AREA_LABELS[a] for a in AREAS]
    if strat["areas"]:
        default_areas = [AREA_LABELS[a] for a in strat["areas"] if a in AREA_LABELS]
    else:
        default_areas = []
    area_sel = st.multiselect(
        "Delivery area(s) — leave empty for all areas",
        area_options, default=default_areas)

    # Hour ranges
    if strat["hour_ranges"]:
        hr_text = ", ".join(f"{h[0]}-{h[1]}" for h in strat["hour_ranges"])
    else:
        hr_text = ""
    hr_input = st.text_input(
        "Hour filter (e.g. '12-15, 17-20'). Leave empty for all hours.",
        value=hr_text)

    # Days
    day_options = ["All days", "Weekdays (Mon-Fri)", "Weekends (Sat-Sun)", "Custom"]
    if strat.get("days") == [0, 1, 2, 3, 4]:
        day_default = 1
    elif strat.get("days") == [5, 6]:
        day_default = 2
    elif strat.get("days"):
        day_default = 3
    else:
        day_default = 0
    day_choice = st.selectbox("Days", day_options, index=day_default)

    # Date filter
    date_options = ["All available dates", "Last 1 week", "Last 2 weeks", "Last 1 month", "Last 2 months"]
    date_days_map = {0: None, 1: 7, 2: 14, 3: 30, 4: 60}
    if strat.get("date_filter_days"):
        dfd = strat["date_filter_days"]
        if dfd <= 7: date_def = 1
        elif dfd <= 14: date_def = 2
        elif dfd <= 30: date_def = 3
        else: date_def = 4
    else:
        date_def = 0
    date_choice = st.selectbox("Date range", date_options, index=date_def)

    return {
        "area_sel": area_sel, "hr_input": hr_input,
        "day_choice": day_choice, "date_choice": date_choice,
        "date_options": date_options, "date_days_map": date_days_map,
    }


def _apply_common_fields(strat: Dict, fields: Dict):
    """Apply form field selections back to the strategy dict."""
    strat["areas"] = [a.split(" ")[0] for a in fields["area_sel"]] if fields["area_sel"] else []

    if fields["hr_input"].strip():
        new_hrs = []
        for part in fields["hr_input"].split(","):
            part = part.strip()
            m = re.match(r"(\d{1,2})\s*[-to]+\s*(\d{1,2})", part)
            if m:
                new_hrs.append((int(m.group(1)), int(m.group(2))))
        strat["hour_ranges"] = new_hrs
    else:
        strat["hour_ranges"] = []

    if fields["day_choice"] == "Weekdays (Mon-Fri)":
        strat["days"] = [0, 1, 2, 3, 4]
    elif fields["day_choice"] == "Weekends (Sat-Sun)":
        strat["days"] = [5, 6]
    elif fields["day_choice"] == "All days":
        strat["days"] = None

    strat["date_filter_days"] = fields["date_days_map"].get(
        fields["date_options"].index(fields["date_choice"]))
    strat["missing"] = []


# ── Main page ───────────────────────────────────────────────────────────────

def main():
    try:
        st.set_page_config(layout="wide", page_title="Strategy Chat")
    except st.errors.StreamlitAPIException:
        pass  # already set by another page
    st.title("Strategy Chat")
    st.markdown(
        "Type a trading strategy **or** a data query in plain English. "
        "Use the **voice input** button for hands-free commands."
    )
    st.markdown(
        "**Datasets:** DayAhead, IDA1, IDA2, IDA3, VWAP &nbsp;|&nbsp; "
        "**Areas:** AT, BE, FR, GER, NL &nbsp;|&nbsp; "
        "**Data:** Jan-Mar 2026 (quarter-hourly)"
    )

    components.html(VOICE_HTML, height=120)

    with st.expander("Example instructions"):
        st.markdown(
            "**Strategy** (buy/sell across datasets):\n"
            "- `Buy dayahead, sell IDA1, weekends, hours 10-12, area GER`\n"
            "- `DA vs IDA1 France and Germany weekends`\n\n"
            "**Compare** (same dataset across areas):\n"
            "- `DA Germany vs France`\n"
            "- `IDA1 GER vs FR vs NL last month`\n\n"
            "**Cross-area** (different datasets, different areas):\n"
            "- `DA France vs IDA1 Germany`\n\n"
            "**Time-shift** (buy/sell same dataset at different hours):\n"
            "- `Buy first 4 hours sell last 4 hours DA Germany`\n"
            "- `Buy off-peak sell peak dayahead France`\n"
            "- `Buy morning sell evening IDA1`\n\n"
            "**Multi-leg volume** (multiple areas with volumes):\n"
            "- `Buy 1MW Germany, 1MW Austria, sell 2MW France DA vs IDA1`\n\n"
            "**Lookup** (view raw data):\n"
            "- `Show DA prices last 30 days for France`\n"
            "- `Full data last month`\n\n"
            "**Follow-up commands** (after results):\n"
            "- `Only Germany` · `Filter weekends` · `Last 7 days`\n"
            "- `Switch to IDA2` · `Exclude France` · `Peak hours only`"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "last_results" not in st.session_state:
        st.session_state.last_results = None      # stored after confirm
        st.session_state.last_results_type = None  # "strategy" or "lookup"
        st.session_state.last_results_strat = None # strategy dict for rendering

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Confirmation / clarification form ───────────────────────────────
    if st.session_state.pending is not None:
        strat = st.session_state.pending
        missing = strat["missing"]
        has_missing = bool(missing)
        mode = strat.get("mode", "strategy")

        if has_missing:
            st.warning("I understood your instruction but need a few more details:")
        else:
            st.info("Here's what I understood. Confirm or adjust before running:")

        if mode == "timeshift":
            # ── Timeshift confirmation form ────────────────────────────────
            with st.form("confirm_timeshift"):
                dataset_key = strat.get("dataset")
                if dataset_key and dataset_key in DATASET_MAP:
                    lf = DATASET_MAP[dataset_key][0]
                    cur = DATASET_DISPLAY.get(lf, dataset_key)
                    ds_idx = DATASET_OPTIONS.index(cur) if cur in DATASET_OPTIONS else 0
                else:
                    ds_idx = 0
                ds_sel = st.selectbox("Dataset", DATASET_OPTIONS, index=ds_idx)

                buy_hours = strat.get("buy_hours", (0, 0))
                sell_hours = strat.get("sell_hours", (0, 0))
                st.markdown(f"**Buy hours:** {buy_hours[0]}:00-{buy_hours[1]}:00")
                st.markdown(f"**Sell hours:** {sell_hours[0]}:00-{sell_hours[1]}:00")

                fields = _render_common_form_fields(strat)
                submitted = st.form_submit_button("Confirm & Run Time-Shift", type="primary")

            if submitted:
                strat["dataset"] = DATASET_KEY_MAP[ds_sel]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                df, error = execute_timeshift(strat)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    lf = DATASET_MAP[strat["dataset"]][0]
                    ds_label = DATASET_DISPLAY.get(lf, strat["dataset"])
                    summary = f"Time-shift: Buy {strat['buy_hours']}, Sell {strat['sell_hours']} on **{ds_label}** ({len(df):,} days)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strat)
                    st.session_state.last_results_type = "timeshift"
                    st.session_state.last_results_strat = strat
                st.rerun()

        elif mode == "multi_leg":
            # ── Multi-leg confirmation form ────────────────────────────────
            with st.form("confirm_multi_leg"):
                # Buy dataset
                if strat.get("buy_dataset") and strat["buy_dataset"] in DATASET_MAP:
                    bf = DATASET_MAP[strat["buy_dataset"]][0]
                    cur = DATASET_DISPLAY.get(bf, strat["buy_dataset"])
                    buy_idx = DATASET_OPTIONS.index(cur) if cur in DATASET_OPTIONS else 0
                else:
                    buy_idx = 0
                buy_sel = st.selectbox("Buy from which dataset?", DATASET_OPTIONS, index=buy_idx)

                # Sell dataset
                if strat.get("sell_dataset") and strat["sell_dataset"] in DATASET_MAP:
                    sf = DATASET_MAP[strat["sell_dataset"]][0]
                    cur = DATASET_DISPLAY.get(sf, strat["sell_dataset"])
                    sell_idx = DATASET_OPTIONS.index(cur) if cur in DATASET_OPTIONS else 1
                else:
                    sell_idx = 1
                sell_sel = st.selectbox("Sell on which dataset?", DATASET_OPTIONS, index=sell_idx)

                st.markdown("**Detected legs:**")
                legs = strat.get("legs", [])
                for leg in legs:
                    st.caption(f"  {leg['side'].upper()} {leg['volume_mw']}MW {leg['area']}")

                fields = _render_common_form_fields(strat)
                submitted = st.form_submit_button("Confirm & Run Multi-Leg", type="primary")

            if submitted:
                strat["buy_dataset"] = DATASET_KEY_MAP[buy_sel]
                strat["sell_dataset"] = DATASET_KEY_MAP[sell_sel]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                df, error = execute_multi_leg(strat)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    legs = strat.get("legs", [])
                    leg_str = ", ".join(f"{l['side'].upper()} {l['volume_mw']}MW {l['area']}" for l in legs)
                    summary = f"Multi-leg: {leg_str} ({len(df):,} slots)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strat)
                    st.session_state.last_results_type = "multi_leg"
                    st.session_state.last_results_strat = strat
                st.rerun()

        elif mode == "lookup":
            # ── Lookup confirmation form ───────────────────────────────
            with st.form("confirm_lookup"):
                if strat.get("lookup_dataset") and strat["lookup_dataset"] in DATASET_MAP:
                    lf = DATASET_MAP[strat["lookup_dataset"]][0]
                    cur = DATASET_DISPLAY.get(lf, strat["lookup_dataset"])
                    ds_idx = DATASET_OPTIONS.index(cur) if cur in DATASET_OPTIONS else 0
                else:
                    ds_idx = 0
                ds_sel = st.selectbox("Dataset", DATASET_OPTIONS, index=ds_idx)

                fields = _render_common_form_fields(strat)
                submitted = st.form_submit_button("Confirm & Show Data", type="primary")

            if submitted:
                strat["lookup_dataset"] = DATASET_KEY_MAP[ds_sel]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                df, error = execute_lookup(strat)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    summary = f"Showing **{len(df):,}** records for **{ds_sel}**"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strat)
                    st.session_state.last_results_type = "lookup"
                    st.session_state.last_results_strat = strat
                st.rerun()

        elif mode == "compare":
            # ── Compare confirmation form ──────────────────────────────
            with st.form("confirm_compare"):
                if strat.get("compare_dataset") and strat["compare_dataset"] in DATASET_MAP:
                    lf = DATASET_MAP[strat["compare_dataset"]][0]
                    cur = DATASET_DISPLAY.get(lf, strat["compare_dataset"])
                    ds_idx = DATASET_OPTIONS.index(cur) if cur in DATASET_OPTIONS else 0
                else:
                    ds_idx = 0
                ds_sel = st.selectbox("Dataset to compare across areas", DATASET_OPTIONS, index=ds_idx)

                # Areas (require at least 2 for compare)
                area_options = [AREA_LABELS[a] for a in AREAS]
                if strat["areas"]:
                    default_areas = [AREA_LABELS[a] for a in strat["areas"] if a in AREA_LABELS]
                else:
                    default_areas = []
                area_sel = st.multiselect(
                    "Select at least 2 areas to compare",
                    area_options, default=default_areas)

                fields = _render_common_form_fields(strat)
                fields["area_sel"] = area_sel  # override
                submitted = st.form_submit_button("Confirm & Compare", type="primary")

            if submitted:
                strat["compare_dataset"] = DATASET_KEY_MAP[ds_sel]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                if len(strat["areas"]) < 2:
                    error = "Please select at least 2 areas for comparison."
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    df, error = execute_compare(strat)
                    if df is None:
                        st.session_state.messages.append({"role": "assistant", "content": error})
                        st.session_state.last_results = None
                    else:
                        areas_str = " vs ".join(strat["areas"])
                        lf = DATASET_MAP[strat["compare_dataset"]][0]
                        ds_label = DATASET_DISPLAY.get(lf, strat["compare_dataset"])
                        summary = f"Comparing **{ds_label}** across **{areas_str}** ({len(df):,} records)"
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                        st.session_state.last_results = (df, strat)
                        st.session_state.last_results_type = "compare"
                        st.session_state.last_results_strat = strat
                st.rerun()

        elif mode == "cross":
            # ── Cross-area confirmation form ───────────────────────────
            with st.form("confirm_cross"):
                buy_file = DATASET_MAP[strat["buy_dataset"]][0] if strat.get("buy_dataset") and strat["buy_dataset"] in DATASET_MAP else None
                buy_disp = DATASET_DISPLAY.get(buy_file, strat.get("buy_dataset", "")) if buy_file else ""
                buy_idx = DATASET_OPTIONS.index(buy_disp) if buy_disp in DATASET_OPTIONS else 0
                buy_sel = st.selectbox("Buy dataset", DATASET_OPTIONS, index=buy_idx)

                area_options_list = [AREA_LABELS[a] for a in AREAS]
                buy_area_default = [i for i, a in enumerate(AREAS) if a == strat.get("buy_area")]
                buy_area_sel = st.selectbox("Buy area", area_options_list,
                                            index=buy_area_default[0] if buy_area_default else 0)

                sell_file = DATASET_MAP[strat["sell_dataset"]][0] if strat.get("sell_dataset") and strat["sell_dataset"] in DATASET_MAP else None
                sell_disp = DATASET_DISPLAY.get(sell_file, strat.get("sell_dataset", "")) if sell_file else ""
                sell_idx = DATASET_OPTIONS.index(sell_disp) if sell_disp in DATASET_OPTIONS else 1
                sell_sel = st.selectbox("Sell dataset", DATASET_OPTIONS, index=sell_idx)

                sell_area_default = [i for i, a in enumerate(AREAS) if a == strat.get("sell_area")]
                sell_area_sel = st.selectbox("Sell area", area_options_list,
                                             index=sell_area_default[0] if sell_area_default else 1)

                fields = _render_common_form_fields(strat)
                submitted = st.form_submit_button("Confirm & Run Cross-Area Strategy", type="primary")

            if submitted:
                strat["buy_dataset"] = DATASET_KEY_MAP[buy_sel]
                strat["buy_area"] = buy_area_sel.split(" ")[0]
                strat["sell_dataset"] = DATASET_KEY_MAP[sell_sel]
                strat["sell_area"] = sell_area_sel.split(" ")[0]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                df, error = execute_cross(strat)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    summary = f"Cross-area: Buy **{buy_sel} ({strat['buy_area']})** → Sell **{sell_sel} ({strat['sell_area']})** ({len(df):,} slots)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strat)
                    st.session_state.last_results_type = "cross"
                    st.session_state.last_results_strat = strat
                st.rerun()

        else:
            # ── Strategy confirmation form ─────────────────────────────
            with st.form("confirm_strategy"):
                # Buy dataset
                if "buy_dataset" in missing or strat["buy_dataset"] is None:
                    buy_sel = st.selectbox("Buy from which dataset?", DATASET_OPTIONS, index=0)
                else:
                    buy_file = DATASET_MAP[strat["buy_dataset"]][0]
                    cur_display = DATASET_DISPLAY.get(buy_file, strat["buy_dataset"])
                    buy_idx = DATASET_OPTIONS.index(cur_display) if cur_display in DATASET_OPTIONS else 0
                    buy_sel = st.selectbox("Buy from which dataset?", DATASET_OPTIONS, index=buy_idx)

                # Sell dataset
                if "sell_dataset" in missing or strat["sell_dataset"] is None:
                    sell_sel = st.selectbox("Sell on which dataset?", DATASET_OPTIONS, index=1)
                else:
                    sell_file = DATASET_MAP[strat["sell_dataset"]][0]
                    cur_display = DATASET_DISPLAY.get(sell_file, strat["sell_dataset"])
                    sell_idx = DATASET_OPTIONS.index(cur_display) if cur_display in DATASET_OPTIONS else 1
                    sell_sel = st.selectbox("Sell on which dataset?", DATASET_OPTIONS, index=sell_idx)

                st.markdown("**Hour ranges** (add strategy variants):")
                fields = _render_common_form_fields(strat)
                submitted = st.form_submit_button("Confirm & Run Strategy", type="primary")

            if submitted:
                strat["buy_dataset"] = DATASET_KEY_MAP[buy_sel]
                strat["sell_dataset"] = DATASET_KEY_MAP[sell_sel]
                _apply_common_fields(strat, fields)
                st.session_state.pending = None

                results, error = execute_multi_strategy(strat)
                if results is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    total_slots = sum(len(df) for df, _ in results)
                    summary = f"Running **{len(results)}** strategy variant(s) across **{total_slots:,}** slots..."
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = results
                    st.session_state.last_results_type = "strategy"
                    st.session_state.last_results_strat = strat
                st.rerun()

    # ── Render stored results (after form collapsed via rerun) ──────────
    if st.session_state.last_results is not None and st.session_state.pending is None:
        if st.session_state.last_results_type == "lookup":
            df, strat_for_render = st.session_state.last_results
            render_lookup(df, strat_for_render)
        elif st.session_state.last_results_type == "lookup_multi":
            df, strat_for_render = st.session_state.last_results
            render_lookup_multi(df, strat_for_render)
        elif st.session_state.last_results_type == "compare":
            df, strat_for_render = st.session_state.last_results
            render_compare(df, strat_for_render)
        elif st.session_state.last_results_type == "cross":
            df, strat_for_render = st.session_state.last_results
            render_cross(df, strat_for_render)
        elif st.session_state.last_results_type == "timeshift":
            df, strat_for_render = st.session_state.last_results
            render_timeshift(df, strat_for_render)
        elif st.session_state.last_results_type == "multi_leg":
            df, strat_for_render = st.session_state.last_results
            render_multi_leg(df, strat_for_render)
        elif st.session_state.last_results_type == "strategy":
            render_multi_results(
                st.session_state.last_results,
                st.session_state.last_results_strat,
            )

    # ── Chat input ──────────────────────────────────────────────────────
    if prompt := st.chat_input("e.g. 'Buy dayahead sell IDA1 France weekends' or 'Show DA prices last 30 days'"):
        # ── Check for follow-up command first ─────────────────────────
        followup = detect_followup(prompt, st.session_state.last_results_strat)
        if followup is not None and st.session_state.last_results is not None:
            # Check for unsupported follow-up (e.g. "change color to green")
            if isinstance(followup, dict) and followup.get("_unsupported_followup"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"I recognized *\"{prompt}\"* as a modification request, but chart style changes "
                        "aren't supported yet. You can modify **areas** (e.g. *only Germany*), "
                        "**days** (*weekends only*), **date range** (*last 7 days*), "
                        "**hours** (*peak hours only*), or **dataset** (*switch to IDA2*)."
                    ),
                })
                st.rerun()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": f"Applying modification: *{prompt}*"})
            _execute_and_store_followup(followup)
            st.rerun()

        # ── New query — clear old data ────────────────────────────────
        st.session_state.messages = []
        st.session_state.last_results = None
        st.session_state.last_results_type = None
        st.session_state.last_results_strat = None

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        strategy = parse_instruction(prompt)
        mode = strategy.get("mode", "strategy")

        # ── Handle unrecognized input gracefully ──────────────────────
        if mode == "unrecognized":
            help_msg = (
                "I couldn't understand that query. Here are some things you can try:\n\n"
                "**View data:** `Show DA prices last 30 days for France`\n\n"
                "**Strategy:** `Buy dayahead sell IDA1 France weekends`\n\n"
                "**Compare areas:** `DA Germany vs France`\n\n"
                "**Time-shift:** `Buy morning sell evening DA Germany`\n\n"
                "Type a command using **datasets** (DA, IDA1, IDA2, IDA3, VWAP) "
                "and **areas** (AT, BE, FR, GER, NL)."
            )
            st.session_state.messages.append({"role": "assistant", "content": help_msg})
            with st.chat_message("assistant"):
                st.markdown(help_msg)
            st.rerun()

        # Build parsed summary
        parsed_parts = []
        if mode == "lookup":
            multi_ds = strategy.get("lookup_datasets")
            if multi_ds and len(multi_ds) >= 2:
                labels = []
                for ds_key in multi_ds:
                    lf = DATASET_MAP[ds_key][0]
                    labels.append(DATASET_DISPLAY.get(lf, ds_key.upper()))
                parsed_parts.append(f"Datasets: **{', '.join(labels)}**")
            elif strategy.get("lookup_dataset"):
                lf = DATASET_MAP[strategy["lookup_dataset"]][0]
                parsed_parts.append(f"Dataset: **{DATASET_DISPLAY.get(lf, strategy['lookup_dataset'])}**")
        elif mode == "compare":
            if strategy.get("compare_dataset"):
                lf = DATASET_MAP[strategy["compare_dataset"]][0]
                parsed_parts.append(f"Dataset: **{DATASET_DISPLAY.get(lf, strategy['compare_dataset'])}**")
            if strategy["areas"]:
                parsed_parts.append(f"Comparing: **{' vs '.join(strategy['areas'])}**")
        elif mode == "cross":
            buy_f = DATASET_MAP[strategy["buy_dataset"]][0]
            sell_f = DATASET_MAP[strategy["sell_dataset"]][0]
            buy_lbl = DATASET_DISPLAY.get(buy_f, strategy["buy_dataset"])
            sell_lbl = DATASET_DISPLAY.get(sell_f, strategy["sell_dataset"])
            parsed_parts.append(f"Buy: **{buy_lbl} ({strategy.get('buy_area', '?')})**")
            parsed_parts.append(f"Sell: **{sell_lbl} ({strategy.get('sell_area', '?')})**")
        elif mode == "timeshift":
            if strategy.get("dataset"):
                lf = DATASET_MAP[strategy["dataset"]][0]
                parsed_parts.append(f"Dataset: **{DATASET_DISPLAY.get(lf, strategy['dataset'])}**")
            buy_h = strategy.get("buy_hours", (0, 0))
            sell_h = strategy.get("sell_hours", (0, 0))
            parsed_parts.append(f"Buy: **{buy_h[0]}:00-{buy_h[1]}:00**")
            parsed_parts.append(f"Sell: **{sell_h[0]}:00-{sell_h[1]}:00**")
        elif mode == "multi_leg":
            if strategy.get("buy_dataset"):
                bf = DATASET_MAP[strategy["buy_dataset"]][0]
                parsed_parts.append(f"Buy: **{DATASET_DISPLAY.get(bf, strategy['buy_dataset'])}**")
            if strategy.get("sell_dataset"):
                sf = DATASET_MAP[strategy["sell_dataset"]][0]
                parsed_parts.append(f"Sell: **{DATASET_DISPLAY.get(sf, strategy['sell_dataset'])}**")
            legs = strategy.get("legs", [])
            if legs:
                leg_str = ", ".join(f"{l['side'].upper()} {l['volume_mw']}MW {l['area']}" for l in legs)
                parsed_parts.append(f"Legs: **{leg_str}**")
        else:
            if strategy["buy_dataset"]:
                bf = DATASET_MAP[strategy["buy_dataset"]][0]
                parsed_parts.append(f"Buy: **{DATASET_DISPLAY.get(bf, strategy['buy_dataset'])}**")
            if strategy["sell_dataset"]:
                sf = DATASET_MAP[strategy["sell_dataset"]][0]
                parsed_parts.append(f"Sell: **{DATASET_DISPLAY.get(sf, strategy['sell_dataset'])}**")
        if mode not in ("compare", "cross", "timeshift"):
            if strategy["areas"]:
                parsed_parts.append(f"Areas: **{', '.join(strategy['areas'])}**")
            elif mode not in ("multi_leg",):
                parsed_parts.append("Areas: **All**")
        if mode not in ("timeshift",) and strategy["hour_ranges"]:
            parsed_parts.append("Hours: **" + " & ".join(
                f"{h[0]}:00-{h[1]}:00" for h in strategy["hour_ranges"]) + "**")
        if strategy["days"]:
            dn = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            parsed_parts.append("Days: **" + ", ".join(dn[d] for d in strategy["days"]) + "**")
        if strategy.get("date_filter_days"):
            parsed_parts.append(f"Date filter: **last {strategy['date_filter_days']} days**")

        mode_labels = {"lookup": "Data lookup", "compare": "Area comparison", "cross": "Cross-area strategy", "timeshift": "Time-shift strategy", "multi_leg": "Multi-leg volume", "strategy": "Strategy"}
        mode_label = mode_labels.get(mode, "Strategy")
        parsed_msg = f"**{mode_label}** — Here's what I understood:\n\n" + " &nbsp;|&nbsp; ".join(parsed_parts)
        if strategy["missing"]:
            missing_labels = {
                "buy_dataset": "buy dataset", "sell_dataset": "sell dataset",
                "areas": "delivery area(s)", "lookup_dataset": "dataset",
            }
            missing_str = ", ".join(missing_labels.get(m, m) for m in strategy["missing"])
            parsed_msg += f"\n\nMissing: **{missing_str}** — please fill in below."

        with st.chat_message("assistant"):
            st.markdown(parsed_msg)
        st.session_state.messages.append({"role": "assistant", "content": parsed_msg})

        # ── Auto-execute if nothing is missing ──────────────────────────
        if not strategy["missing"]:
            if mode == "lookup":
                # Multi-dataset lookup: run each dataset separately and show them all
                multi_ds = strategy.get("lookup_datasets")
                if multi_ds and len(multi_ds) >= 2:
                    all_frames = []
                    ds_labels = []
                    for ds_key in multi_ds:
                        sub_params = dict(strategy)
                        sub_params["lookup_dataset"] = ds_key
                        sub_df, sub_err = execute_lookup(sub_params)
                        if sub_df is not None and not sub_df.empty:
                            lf = DATASET_MAP[ds_key][0]
                            ds_label = DATASET_DISPLAY.get(lf, ds_key.upper())
                            sub_df = sub_df.copy()
                            sub_df["_dataset"] = ds_label
                            all_frames.append(sub_df)
                            ds_labels.append(ds_label)
                    if all_frames:
                        combined_df = pd.concat(all_frames, ignore_index=True)
                        labels_str = ", ".join(ds_labels)
                        summary = f"Showing **{len(combined_df):,}** records across **{labels_str}**"
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                        st.session_state.last_results = (combined_df, strategy)
                        st.session_state.last_results_type = "lookup_multi"
                        st.session_state.last_results_strat = strategy
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "No data found for the requested datasets."})
                        st.session_state.last_results = None
                else:
                    df, error = execute_lookup(strategy)
                    if df is None:
                        st.session_state.messages.append({"role": "assistant", "content": error})
                        st.session_state.last_results = None
                    else:
                        lf = DATASET_MAP[strategy["lookup_dataset"]][0]
                        ds_label = DATASET_DISPLAY.get(lf, strategy["lookup_dataset"])
                        summary = f"Showing **{len(df):,}** records for **{ds_label}**"
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                        st.session_state.last_results = (df, strategy)
                        st.session_state.last_results_type = "lookup"
                        st.session_state.last_results_strat = strategy
                st.rerun()
            elif mode == "compare":
                df, error = execute_compare(strategy)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    areas_str = " vs ".join(strategy["areas"])
                    lf = DATASET_MAP[strategy["compare_dataset"]][0]
                    ds_label = DATASET_DISPLAY.get(lf, strategy["compare_dataset"])
                    summary = f"Comparing **{ds_label}** across **{areas_str}** ({len(df):,} records)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strategy)
                    st.session_state.last_results_type = "compare"
                    st.session_state.last_results_strat = strategy
                st.rerun()
            elif mode == "cross":
                df, error = execute_cross(strategy)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    buy_f = DATASET_MAP[strategy["buy_dataset"]][0]
                    sell_f = DATASET_MAP[strategy["sell_dataset"]][0]
                    buy_lbl = DATASET_DISPLAY.get(buy_f, strategy["buy_dataset"])
                    sell_lbl = DATASET_DISPLAY.get(sell_f, strategy["sell_dataset"])
                    summary = (f"Cross-area: Buy **{buy_lbl} ({strategy['buy_area']})** → "
                               f"Sell **{sell_lbl} ({strategy['sell_area']})** ({len(df):,} slots)")
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strategy)
                    st.session_state.last_results_type = "cross"
                    st.session_state.last_results_strat = strategy
                st.rerun()
            elif mode == "timeshift":
                df, error = execute_timeshift(strategy)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    lf = DATASET_MAP[strategy["dataset"]][0]
                    ds_label = DATASET_DISPLAY.get(lf, strategy["dataset"])
                    buy_h = strategy.get("buy_hours", (0, 0))
                    sell_h = strategy.get("sell_hours", (0, 0))
                    summary = f"Time-shift: Buy {buy_h}, Sell {sell_h} on **{ds_label}** ({len(df):,} days)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strategy)
                    st.session_state.last_results_type = "timeshift"
                    st.session_state.last_results_strat = strategy
                st.rerun()
            elif mode == "multi_leg":
                df, error = execute_multi_leg(strategy)
                if df is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    legs = strategy.get("legs", [])
                    leg_str = ", ".join(f"{l['side'].upper()} {l['volume_mw']}MW {l['area']}" for l in legs)
                    summary = f"Multi-leg: {leg_str} ({len(df):,} slots)"
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = (df, strategy)
                    st.session_state.last_results_type = "multi_leg"
                    st.session_state.last_results_strat = strategy
                st.rerun()
            else:
                results, error = execute_multi_strategy(strategy)
                if results is None:
                    st.session_state.messages.append({"role": "assistant", "content": error})
                    st.session_state.last_results = None
                else:
                    total_slots = sum(len(df) for df, _ in results)
                    summary = f"Running **{len(results)}** strategy variant(s) across **{total_slots:,}** slots..."
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    st.session_state.last_results = results
                    st.session_state.last_results_type = "strategy"
                    st.session_state.last_results_strat = strategy
                st.rerun()
        else:
            st.session_state.pending = strategy
            st.rerun()


main()
