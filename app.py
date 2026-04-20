# ============================================================
# 🌐 Multilingual AI Translator — Powered by Groq
# ============================================================
# VERSION 3.1  — Gradio 6.0 Compatible (HF Spaces)
#
# FIXES from v3.0:
#   ✅ theme, css, js moved from gr.Blocks() → demo.launch()
#   ✅ show_copy_button removed (deprecated in Gradio 6.0)
#   ✅ show_progress="hidden"/"minimal" → show_progress=False/True
#   ✅ gr.close_all() removed (not needed on HF Spaces)
#   ✅ server_port=7860, share=False for HF Spaces
#
# FEATURES:
#   ✅ PHASE 1: Bidirectional translation + Download .txt
#   ✅ PHASE 2: Text-to-Speech + File Upload
#   ✅ PHASE 3: Analytics Tab + About Tab
#   ✅ PHASE 4: 8 languages (Urdu, English, Punjabi, Sindhi,
#               Pashto, Arabic, French, German)
# ============================================================

import os
import re
import tempfile
import gradio as gr
from datetime import datetime
from groq import Groq, APIError, AuthenticationError, RateLimitError

# ── Model configuration ──────────────────────────────────────────
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ── Supported languages ──────────────────────────────────────────
LANGUAGES = [
    "Urdu", "English", "Punjabi", "Sindhi", "Pashto", "Arabic", "French", "German"
]


# ════════════════════════════════════════════════════════════════
# 1. Groq client
# ════════════════════════════════════════════════════════════════
def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEY is not set.\n"
            "On HF Spaces: Settings → Variables and secrets → add GROQ_API_KEY."
        )
    return Groq(api_key=api_key)


# ════════════════════════════════════════════════════════════════
# 2. Urdu script / Romanized detector
# ════════════════════════════════════════════════════════════════
def detect_input_type(text: str) -> str:
    if re.search(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text):
        return "urdu_script"
    roman_words = {
        "kia", "kya", "hai", "ha", "hain", "mein", "main", "se", "ko", "ka",
        "ki", "ke", "aur", "nahi", "nahin", "ap", "aap", "tum", "hum", "yeh",
        "woh", "kahan", "kab", "kyun", "kyoun", "tha", "thi", "the", "kal",
        "aj", "aaj", "raat", "din", "log", "baat", "dil", "zindagi", "mosam",
        "mosum", "acha", "accha", "theek", "shukriya", "meherbani", "zaroor",
        "bohat", "bahut", "bilkul", "haan", "nai", "liye", "saath", "kuch",
        "sab", "phir", "abhi", "sirf", "lekin", "magar", "kyunke", "agr",
        "agar", "toh", "to", "wala", "wali", "bhook", "pani", "khana",
    }
    tokens = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    if tokens & roman_words:
        return "romanized"
    return "other"


# ════════════════════════════════════════════════════════════════
# 3. Core translation
# ════════════════════════════════════════════════════════════════
def translate(
    input_text: str,
    source_lang: str,
    target_lang: str,
    formality: str,
    history: list,
    analytics: dict,
) -> tuple:
    if not input_text or not input_text.strip():
        return "⚠️ Please enter some text.", history, "", analytics, None

    if source_lang == target_lang:
        return "⚠️ Source and target languages are the same!", history, "", analytics, None

    if len(input_text) > 8_000:
        return (
            f"⚠️ Input too long ({len(input_text):,} chars). Max 8,000.",
            history, "", analytics, None
        )

    badge = f"🌐 {source_lang} → {target_lang}"
    if source_lang == "Urdu":
        itype = detect_input_type(input_text.strip())
        if itype == "urdu_script":
            badge = "🔤 Urdu script → " + target_lang
        elif itype == "romanized":
            badge = "🔡 Romanized Urdu → " + target_lang

    tone_map = {
        "Casual":  "Use a casual, conversational tone.",
        "Neutral": "Use a neutral, everyday tone.",
        "Formal":  "Use a formal, professional tone.",
    }

    if source_lang == "Urdu" and detect_input_type(input_text.strip()) == "romanized":
        system_prompt = (
            f"You are an expert translator. "
            f"The input is Romanized Urdu (Urdu words in Latin letters, e.g. 'kia hal ha'). "
            f"Translate it to {target_lang}. "
            f"{tone_map.get(formality, '')} "
            f"Return ONLY the translated text — no notes, no explanations."
        )
    else:
        system_prompt = (
            f"You are an expert translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"{tone_map.get(formality, '')} "
            f"Return ONLY the translated text — no notes, no explanations."
        )

    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            client   = get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": input_text.strip()},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            result = response.choices[0].message.content.strip()

            if not result:
                return "⚠️ Model returned empty output.", history, "", analytics, None

            words = len(result.split())
            sents = max(len(re.findall(r'[.!?]+', result)), 1)
            stats = (
                f"{badge}  ·  📊 **{words} words, {sents} sentence{'s' if sents>1 else ''}**"
                f"  ·  🤖 `{model}`  ·  🎚️ _{formality}_"
            )

            short    = (input_text[:55] + "…") if len(input_text) > 55 else input_text
            new_hist = [[short, result, source_lang, target_lang]] + (history or [])
            new_hist = new_hist[:5]

            new_analytics = analytics.copy()
            new_analytics["total"]       = analytics.get("total", 0) + 1
            new_analytics["total_words"] = analytics.get("total_words", 0) + words
            pair  = f"{source_lang}→{target_lang}"
            pairs = new_analytics.get("pairs", {})
            pairs[pair] = pairs.get(pair, 0) + 1
            new_analytics["pairs"] = pairs
            tones = new_analytics.get("tones", {})
            tones[formality] = tones.get(formality, 0) + 1
            new_analytics["tones"] = tones

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmpfile   = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt",
                prefix=f"translation_{timestamp}_",
                mode="w", encoding="utf-8"
            )
            tmpfile.write(
                f"=== Translation Result ===\n"
                f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"From     : {source_lang}\n"
                f"To       : {target_lang}\n"
                f"Tone     : {formality}\n"
                f"Model    : {model}\n"
                f"{'='*40}\n\n"
                f"[Original]\n{input_text.strip()}\n\n"
                f"[Translation]\n{result}\n"
            )
            tmpfile.close()

            return result, new_hist, stats, new_analytics, tmpfile.name

        except AuthenticationError:
            return ("🔑 Auth failed. Check GROQ_API_KEY at console.groq.com",
                    history, "", analytics, None)
        except RateLimitError:
            return ("⏳ Rate limit hit. Wait ~60 seconds.",
                    history, "", analytics, None)
        except APIError as e:
            err_str = str(e).lower()
            if "decommissioned" in err_str or "model_not_found" in err_str:
                continue
            return (f"🚨 Groq API error: {e.message}", history, "", analytics, None)
        except ValueError as e:
            return (str(e), history, "", analytics, None)
        except Exception as e:
            return (f"🚨 Unexpected error: {str(e)}", history, "", analytics, None)

    return ("🚨 Both models unavailable. Try again later.", history, "", analytics, None)


# ════════════════════════════════════════════════════════════════
# 4. File upload translation
# ════════════════════════════════════════════════════════════════
def translate_file(file, source_lang, target_lang, formality, history, analytics):
    if file is None:
        return "⚠️ No file uploaded.", history, "", analytics, None
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8_000:
            content = content[:8_000]
        return translate(content, source_lang, target_lang, formality, history, analytics)
    except Exception as e:
        return (f"🚨 File read error: {str(e)}", history, "", analytics, None)


# ════════════════════════════════════════════════════════════════
# 5. History display
# ════════════════════════════════════════════════════════════════
def format_history(hist: list) -> str:
    if not hist:
        return "_Your last 5 translations will appear here._"
    rows = []
    for i, item in enumerate(hist, 1):
        src, tgt, sl, tl = item if len(item) == 4 else (item[0], item[1], "?", "?")
        rows.append(f"**{i}.** `{src}` _{sl}→{tl}_\n→ {tgt}")
    return "\n\n---\n\n".join(rows)


# ════════════════════════════════════════════════════════════════
# 6. Analytics display
# ════════════════════════════════════════════════════════════════
def format_analytics(a: dict) -> str:
    if not a or a.get("total", 0) == 0:
        return "_No translations yet this session._"

    total = a.get("total", 0)
    words = a.get("total_words", 0)
    avg   = round(words / total) if total else 0
    pairs = a.get("pairs", {})
    tones = a.get("tones", {})

    top_pair = max(pairs, key=pairs.get) if pairs else "—"
    top_tone = max(tones, key=tones.get) if tones else "—"

    pair_rows = "\n".join(
        f"  - `{k}`: {v} translation{'s' if v>1 else ''}"
        for k, v in sorted(pairs.items(), key=lambda x: -x[1])
    )
    tone_rows = "\n".join(
        f"  - {k}: {v}×"
        for k, v in sorted(tones.items(), key=lambda x: -x[1])
    )

    return (
        f"### 📊 Session Analytics\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Translations | **{total}** |\n"
        f"| Total Words Translated | **{words:,}** |\n"
        f"| Avg Words per Translation | **{avg}** |\n"
        f"| Most Used Pair | **{top_pair}** |\n"
        f"| Most Used Tone | **{top_tone}** |\n\n"
        f"**Language Pairs:**\n{pair_rows}\n\n"
        f"**Tone Breakdown:**\n{tone_rows}"
    )


# ════════════════════════════════════════════════════════════════
# 7. Character counter
# ════════════════════════════════════════════════════════════════
def count_chars(text: str) -> str:
    n    = len(text) if text else 0
    icon = "🟢" if n < 6_000 else ("🟡" if n < 8_000 else "🔴")
    return f"{icon} {n:,} / 8,000 characters"


# ════════════════════════════════════════════════════════════════
# 8. TTS JavaScript  (injected via demo.launch js= in Gradio 6)
# ════════════════════════════════════════════════════════════════
TTS_JS = """
function speakText() {
    const boxes = document.querySelectorAll('#output-box textarea');
    const text  = boxes.length ? boxes[0].value : '';
    if (!text || !text.trim()) { alert('No translation to speak yet!'); return; }
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate  = 0.9;
    utt.pitch = 1.0;
    window.speechSynthesis.speak(utt);
}
function stopSpeak() { window.speechSynthesis.cancel(); }
"""

# ════════════════════════════════════════════════════════════════
# 9. CSS
# ════════════════════════════════════════════════════════════════
CSS = """
    .gradio-container { max-width: 1100px !important; margin: auto; }

    #header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
        border-radius: 20px; padding: 30px 36px; margin-bottom: 16px;
        box-shadow: 0 10px 40px rgba(15,76,117,.5);
        border: 1px solid rgba(99,179,237,.15);
    }
    #header h1 {
        font-size: 2.1rem; font-weight: 900; color: #e0f2fe !important;
        margin: 0 0 6px; letter-spacing: -1px;
    }
    #header p { color: #93c5fd !important; margin: 0; font-size: .93rem; line-height: 1.6; }

    .panel {
        border: 1px solid rgba(99,179,237,.2); border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,.08); padding: 8px;
    }

    textarea { font-size: 1.05rem !important; line-height: 1.8 !important; }
    #input-box  textarea { font-family: 'Noto Nastaliq Urdu', 'Segoe UI', sans-serif !important; }
    #output-box textarea {
        background: #f0f9ff !important; color: #0c4a6e !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    #translate-btn {
        background: linear-gradient(135deg, #0ea5e9, #0369a1) !important;
        border: none !important; color: white !important; font-weight: 800 !important;
        font-size: 1rem !important; border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(14,165,233,.4) !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    #translate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(14,165,233,.55) !important;
    }
    #tts-btn {
        background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
        border: none !important; color: white !important;
        font-weight: 700 !important; border-radius: 12px !important;
    }
    #stop-btn {
        background: linear-gradient(135deg, #dc2626, #991b1b) !important;
        border: none !important; color: white !important;
        font-weight: 700 !important; border-radius: 12px !important;
    }
    #clear-btn { border-radius: 12px !important; font-weight: 600 !important; }

    #char-counter { font-size: .82rem; color: #64748b; text-align: right; margin-top: 2px; }
    #stats-bar    { font-size: .80rem; color: #475569; margin-top: 4px; padding: 4px 2px; }
    #footer       { text-align: center; color: #94a3b8; font-size: .78rem;
                    margin-top: 16px; padding: 12px; }
    .hist-box {
        background: #f8fafc; border-radius: 12px;
        padding: 12px 16px; font-size: .88rem; line-height: 1.7;
    }
"""

# ════════════════════════════════════════════════════════════════
# 10. UI  — Gradio 6.0: theme/css/js go to launch(), NOT Blocks()
# ════════════════════════════════════════════════════════════════
with gr.Blocks(title="🌐 Multilingual AI Translator") as demo:

    history_state   = gr.State([])
    analytics_state = gr.State({"total": 0, "total_words": 0, "pairs": {}, "tones": {}})

    # Header
    with gr.Column(elem_id="header"):
        gr.HTML(
            "<h1>🌐 Multilingual AI Translator</h1>"
            "<p><b>Powered by Groq · LLaMA 3.3 70B</b> &nbsp;|&nbsp; "
            "Urdu 🇵🇰 · Punjabi · Sindhi · Pashto · "
            "Arabic 🇸🇦 · English 🇬🇧 · French 🇫🇷 · German 🇩🇪"
            " &nbsp;|&nbsp; TTS · File Upload · Download · Analytics</p>"
        )

    with gr.Tabs():

        # ── TAB 1: Translator ────────────────────────────────
        with gr.Tab("🔤 Translator"):

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=LANGUAGES, value="Urdu",
                    label="📥 Source Language", scale=2,
                )
                gr.HTML(
                    "<div style='display:flex;align-items:center;"
                    "padding:20px 8px 0;font-size:1.6rem;'>⇄</div>"
                )
                target_lang = gr.Dropdown(
                    choices=LANGUAGES, value="English",
                    label="📤 Target Language", scale=2,
                )
                formality = gr.Radio(
                    choices=["Casual", "Neutral", "Formal"],
                    value="Neutral", label="🎚️ Tone", scale=3,
                )

            with gr.Row(equal_height=True):

                # LEFT — Input
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("### ✍️ Input Text")
                    input_box = gr.Textbox(
                        label="",
                        placeholder=(
                            "Type in any supported language…\n\n"
                            "Urdu script:    یہاں اردو لکھیں\n"
                            "Romanized Urdu: kia hal ha? bohat acha din hai.\n"
                            "English:        Type your English text here."
                        ),
                        lines=12, max_lines=30,
                        elem_id="input-box",
                    )
                    char_counter = gr.Markdown(
                        "🟢 0 / 8,000 characters", elem_id="char-counter"
                    )
                    with gr.Row():
                        clear_btn     = gr.Button("🗑️ Clear",     variant="secondary",
                                                  elem_id="clear-btn",     scale=1)
                        translate_btn = gr.Button("🚀 Translate", variant="primary",
                                                  elem_id="translate-btn", scale=3)

                # RIGHT — Output
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("### 🌍 Translation Output")
                    output_box = gr.Textbox(
                        label="",
                        placeholder="Your translation will appear here…",
                        lines=12, max_lines=30,
                        interactive=False,
                        elem_id="output-box",
                    )
                    stats_bar = gr.Markdown("", elem_id="stats-bar")
                    with gr.Row():
                        tts_btn  = gr.Button("🔊 Speak", variant="secondary",
                                             elem_id="tts-btn",  scale=2)
                        stop_btn = gr.Button("⏹ Stop",  variant="secondary",
                                             elem_id="stop-btn", scale=1)
                    dl_file = gr.File(label="⬇️ Download Translation (.txt)")

            # File upload
            with gr.Accordion("📂 Upload .txt File for Bulk Translation", open=False):
                gr.Markdown("Upload a `.txt` file (max 8,000 chars) to translate its content.")
                file_upload        = gr.File(
                    label="Choose .txt file", file_types=[".txt"], type="filepath"
                )
                file_translate_btn = gr.Button("📄 Translate File", variant="primary")

            # Examples
            gr.Examples(
                label="💡 Quick Examples — click any to load",
                examples=[
                    ["kia hal ha?"],
                    ["aaj mosam bohat acha hai, dhoop nikli hui hai."],
                    ["پاکستان جنوبی ایشیا میں واقع ایک خوبصورت ملک ہے۔"],
                    ["تعلیم ہر انسان کا بنیادی حق ہے اور اسے سب کے لیے قابل رسائی ہونا چاہیے۔"],
                    ["Education is the most powerful weapon you can use to change the world."],
                    ["La vie est belle et pleine de surprises magnifiques."],
                ],
                inputs=input_box,
            )

            # History
            with gr.Accordion("🕘 Translation History (last 5)", open=False):
                history_md = gr.Markdown(
                    "_Your last 5 translations will appear here._",
                    elem_classes="hist-box",
                )

        # ── TAB 2: Analytics ─────────────────────────────────
        with gr.Tab("📊 Analytics"):
            gr.Markdown(
                "## 📊 Session Analytics\n"
                "Updates automatically after every translation."
            )
            analytics_md = gr.Markdown("_No translations yet this session._")
            refresh_btn  = gr.Button("🔄 Refresh Stats", variant="secondary")

        # ── TAB 3: About ─────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.HTML("""
            <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
                        border-radius:14px;padding:22px 28px;margin:8px 0;
                        border:1px solid rgba(14,165,233,.2);">
                <h2 style="margin-top:0;color:#0c4a6e;">🌐 Multilingual AI Translator v3.1</h2>
                <p style="color:#334155;">
                    A full-featured, professional-grade translation app built as a
                    <b>portfolio project</b> demonstrating real-world AI integration.
                </p>
            </div>

            <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
                        border-radius:14px;padding:22px 28px;margin:12px 0;
                        border:1px solid rgba(14,165,233,.2);">
                <h3 style="color:#0c4a6e;margin-top:0;">🛠️ Tech Stack</h3>
                <table style="width:100%;border-collapse:collapse;font-size:.92rem;">
                    <tr style="background:rgba(14,165,233,.1);">
                        <th style="padding:8px 12px;text-align:left;">Component</th>
                        <th style="padding:8px 12px;text-align:left;">Technology</th>
                        <th style="padding:8px 12px;text-align:left;">Purpose</th>
                    </tr>
                    <tr><td style="padding:8px 12px;"><b>LLM</b></td>
                        <td>LLaMA 3.3 70B</td><td>Core translation engine</td></tr>
                    <tr style="background:#f8fafc;">
                        <td style="padding:8px 12px;"><b>API</b></td>
                        <td>Groq Cloud API</td><td>Ultra-fast LLM inference</td></tr>
                    <tr><td style="padding:8px 12px;"><b>UI</b></td>
                        <td>Gradio 6.x</td><td>Interactive web interface</td></tr>
                    <tr style="background:#f8fafc;">
                        <td style="padding:8px 12px;"><b>Language</b></td>
                        <td>Python 3.10+</td><td>Backend logic</td></tr>
                    <tr><td style="padding:8px 12px;"><b>TTS</b></td>
                        <td>Web Speech API (JS)</td><td>Text-to-speech output</td></tr>
                    <tr style="background:#f8fafc;">
                        <td style="padding:8px 12px;"><b>Fallback</b></td>
                        <td>LLaMA 3.1 8B Instant</td><td>Backup model</td></tr>
                    <tr><td style="padding:8px 12px;"><b>Platform</b></td>
                        <td>Hugging Face Spaces</td><td>Cloud deployment</td></tr>
                </table>
            </div>

            <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
                        border-radius:14px;padding:22px 28px;margin:12px 0;
                        border:1px solid rgba(14,165,233,.2);">
                <h3 style="color:#0c4a6e;margin-top:0;">✨ Features</h3>
                <ul style="color:#334155;line-height:2.1;">
                    <li>🔄 <b>Bidirectional translation</b> across 8 languages</li>
                    <li>🔡 <b>Romanized Urdu auto-detection</b></li>
                    <li>🎚️ <b>Tone control</b> — Casual / Neutral / Formal</li>
                    <li>🔊 <b>Text-to-Speech</b> — hear your translation</li>
                    <li>📂 <b>File upload</b> — translate entire .txt files</li>
                    <li>⬇️ <b>Download</b> — save as .txt with metadata</li>
                    <li>📊 <b>Session analytics</b> — track usage stats</li>
                    <li>🕘 <b>Translation history</b> — last 5 remembered</li>
                    <li>🤖 <b>Auto model fallback</b> — never fails silently</li>
                </ul>
            </div>

            <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
                        border-radius:14px;padding:22px 28px;margin:12px 0;
                        border:1px solid rgba(14,165,233,.2);">
                <h3 style="color:#0c4a6e;margin-top:0;">🌍 Supported Languages</h3>
                <p style="color:#334155;font-size:.95rem;line-height:2.2;">
                    🇵🇰 <b>Urdu</b> (script + Romanized) &nbsp;·&nbsp;
                    🇬🇧 <b>English</b> &nbsp;·&nbsp;
                    🟠 <b>Punjabi</b> &nbsp;·&nbsp;
                    🔵 <b>Sindhi</b> &nbsp;·&nbsp;
                    🟢 <b>Pashto</b> &nbsp;·&nbsp;
                    🇸🇦 <b>Arabic</b> &nbsp;·&nbsp;
                    🇫🇷 <b>French</b> &nbsp;·&nbsp;
                    🇩🇪 <b>German</b>
                </p>
            </div>

            <div style="text-align:center;color:#94a3b8;font-size:.8rem;margin-top:20px;">
                Built with ❤️ using <b>Gradio 6 + Groq (LLaMA 3.3 70B)</b> · Portfolio v3.1
            </div>
            """)

    # Footer
    gr.HTML(
        "<div id='footer'>"
        "🌐 <b>Multilingual AI Translator v3.1</b> &nbsp;·&nbsp; "
        "Groq · LLaMA 3.3 70B · Gradio 6 · Python"
        "</div>"
    )

    # ════════════════════════════════════════════════════════
    # 11. Event Handlers
    # ════════════════════════════════════════════════════════

    input_box.change(
        fn=count_chars, inputs=input_box,
        outputs=char_counter, show_progress=False
    )

    def on_translate(text, src, tgt, tone, hist, analytics):
        result, new_hist, stats, new_an, dl_path = translate(
            text, src, tgt, tone, hist, analytics
        )
        return (
            result, new_hist, stats, new_an,
            format_history(new_hist), format_analytics(new_an), dl_path,
        )

    _outputs = [output_box, history_state, stats_bar,
                analytics_state, history_md, analytics_md, dl_file]
    _inputs  = [input_box, source_lang, target_lang,
                formality, history_state, analytics_state]

    translate_btn.click(fn=on_translate, inputs=_inputs, outputs=_outputs)
    input_box.submit(fn=on_translate,    inputs=_inputs, outputs=_outputs)

    def on_file_translate(file, src, tgt, tone, hist, analytics):
        result, new_hist, stats, new_an, dl_path = translate_file(
            file, src, tgt, tone, hist, analytics
        )
        return (
            result, new_hist, stats, new_an,
            format_history(new_hist), format_analytics(new_an), dl_path,
        )

    file_translate_btn.click(
        fn=on_file_translate,
        inputs=[file_upload, source_lang, target_lang,
                formality, history_state, analytics_state],
        outputs=_outputs,
    )

    # TTS via JS
    tts_btn.click(fn=None,  js="() => speakText()")
    stop_btn.click(fn=None, js="() => stopSpeak()")

    def on_clear():
        return ("", "🟢 0 / 8,000 characters", "", [],
                "", "_Your last 5 translations will appear here._", None)

    clear_btn.click(
        fn=on_clear, inputs=None,
        outputs=[input_box, char_counter, output_box,
                 history_state, stats_bar, history_md, dl_file],
    )

    refresh_btn.click(fn=format_analytics, inputs=analytics_state, outputs=analytics_md)


# ════════════════════════════════════════════════════════════════
# 12. Launch  — Gradio 6.0: theme/css/js passed here
# ════════════════════════════════════════════════════════════════
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
    theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
    ),
    css=CSS,
    js=f"() => {{ {TTS_JS} }}",
)
