"""Bundled text normaliser (maya3.text): CPU only, standard library only.

    python -m pytest maya3/tests/test_text.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

import maya3.text as T
from maya3.text import normalise_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("text,lang,want", [
    ("I have 25 apples", "en", "I have twenty five apples"),
    ("मेरे पास 25 रुपये हैं", "hi", "मेरे पास पच्चीस रुपये हैं"),
    ("నా దగ్గర 25 రూపాయలు ఉన్నాయి", "te", "నా దగ్గర ఇరవై ఐదు రూపాయలు ఉన్నాయి"),
    ("Your OTP is 482913.", "en", "Your OTP is four, eight, two, nine, one, three."),
    ("आपका ओटीपी 4821 है", "hi", "आपका ओटीपी चार, आठ, दो, एक है"),
    ("Visit https://www.example.com/abc today", "en", "Visit W W W dot example dot com slash abc today"),
    ("Email a.b@gmail.com", "en", "Email A dot B at gmail dot com"),
    ("Ha [laughs] I have 3 cats", "en", "Ha [laughs] I have three cats"),
    ("मेरे पास 25 रुपये हैं [laughs] और 3 किताबें", "hi", "मेरे पास पच्चीस रुपये हैं [laughs] और तीन किताबें"),
    ("  spaced   [sighs]  out 12%  ", "ta", "spaced [sighs] out பன்னிரண்டு சதவீதம்"),
    ("২৫ টাকা", "bn", "পঁচিশ টাকা"),
])
def test_normalise(text, lang, want):
    assert normalise_text(text, lang) == want


def test_caps_fold_and_acronyms():
    assert normalise_text("RAHUL KUMAR paid", "en") == "Rahul Kumar paid"
    assert normalise_text("HDFC NEFT KYC done", "en") == "HDFC NEFT KYC done"     # vowel-poor: stay acronyms
    assert T._caps_fold("ASHA OTP NASA KYC") == "Asha OTP Nasa KYC"     # 4 letters + 2 vowels -> a name


def test_tags_are_protected():
    out = normalise_text("[laughs] 2 [sighs] [whispers] 3 [laughs]", "en")
    assert out == "[laughs] two [sighs] [whispers] three [laughs]"
    assert normalise_text("[laughs]", "en") == "[laughs]"


def test_fallback_returns_text_unchanged(monkeypatch):
    def boom(s, lang):
        raise RuntimeError("tn failure")
    monkeypatch.setattr(T, "_TN", boom)
    assert normalise_text("I have 25 apples [laughs] 3", "en") == "I have 25 apples [laughs] 3"
    monkeypatch.setattr(T, "_TN", None)
    assert normalise_text("x 1", "fr") == "x 1"                 # unsupported language: unchanged


def test_all_languages_run():
    for lang in ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"):
        out = normalise_text("Pay ₹1,250 on 12/03/2024 at 10:30, call 9820451120 [laughs]", lang)
        assert out and not any(ch.isdigit() for ch in out) and out.endswith("[laughs]"), (lang, out)


def test_vendored_copy_is_verbatim():
    import hashlib
    want = {   # sha256 of the vendored upstream tn files (upstream commit fe5e6ec)
        "assamese_numbers.py": "0f590e4132c5ee539d4fffdf098c58ae1b72ba28c9b939f78ffb78374c043d48",
        "identifier_rules.py": "c3021145c5020b85d8597f4e47415e0341c1baafa0fbc0a80318a8b27f523682",
        "indic_languages.py": "af9195e44e739bf72b2522a253d69edd0d6fc5ff1dbb7f1b5040f548cabcd800",   # one comment line reworded for the public release
        "normalize.py": "b232015a7c02b8ab9b28f5e21571a0cd6c4ad7ee35ab9381bd5ee56a59169b8b",
        "script_guard.py": "5b7405de2997932ff8785da20fbd3c89e5316ce3854638194dec3f4988e019fb",
        "text_pipeline.py": "259b2ac510406bdef6a5f9bc14ebc2d3c28173729bdb6dff71e25ea31b44680a",
        "tts_chunker.py": "6ad313fa5bff97512d07db5164664802c9c9f887dd835e8ad5bc8d814657ff58",
    }
    d = os.path.join(os.path.dirname(T.__file__), "tn")
    got = {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in want}
    assert got == want


def test_imports_with_stdlib_only():
    code = ("import sys; import maya3.text as t; assert t.normalise_text('25', 'hi') == 'पच्चीस'; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'numpy', 'torch', 'scipy', 'aiohttp'}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


GUARDS = os.environ.get("MAYA3_REF_GUARDS", "")   # optional: reference wrapper for the parity test


@pytest.mark.skipif(not os.path.exists(GUARDS), reason="reference eval wrapper not available")
def test_parity_with_eval_wrapper():
    """Same output as the wrapper the release evals used, on the source repo's audit cases in every language."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("_eval_guards", GUARDS)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    base = os.environ.get("MAYA3_REF_NORM_CASES", "")
    cases = [c["input"] for c in json.load(open(base))["cases"]] if os.path.exists(base) else []
    cases += ["Your OTP is 482913 [laughs]", "Visit https://www.example.com/abc", "RAHUL KUMAR paid ₹1,250"]
    for text in cases:
        for lang in ("en", "hi", "te", "ta", "bn", "kn"):
            t = text + " [laughs] " + text.upper()
            assert normalise_text(t, lang, rules=False) == g.normalise_text(t, lang), (lang, t)


# ---------------------------------------------------------------------------------------------------------------
# Residual-failure rules (maya3.text.rules), cases taken from eval rows that still failed with normalisation on.
@pytest.mark.parametrize("text,lang,want", [
    # R1 two-digit-year dates (strict normaliser raised ambiguous_year and read "one five minus zero eight ...")
    ("कृपया लक्षात घ्या की तुमची मुदत ठेव 15-08-25 ला परिपक्व होत आहे", "mr", "कृपया लक्षात घ्या की तुमची मुदत ठेव पंधरा ऑगस्ट दोन हजार पंचवीस ला परिपक्व होत आहे"),
    ("শেষ তারিখ 15-08-25 নির্ধারণ", "bn", "শেষ তারিখ পনেরো আগস্ট দুই হাজার পঁচিশ নির্ধারণ"),
    ("ট্রানজ্যাকশন 9/9/99 তারিখে", "bn", "ট্রানজ্যাকশন নয় সেপ্টেম্বর উনিশশো নিরানব্বই তারিখে"),
    ("12-05-24 ତାରିଖରେ", "or", "ବାର ମଇ ଦୁଇ ହଜାର ଚବିଶି ତାରିଖରେ"),
    ("starting from 05/11/24.", "en", "starting from five November two thousand twenty four."),
    ("കാലാവധി 15-08-25 ന്", "ml", "കാലാവധി പതിനഞ്ച് ഓഗസ്റ്റ് രണ്ടായിരത്തി ഇരുപത്തിയഞ്ചിന്"),
    # R2 card expiry MM/YY next to an expiry / date word or a CVV
    ("ಕಾರ್ಡ್‌ನ ಅಂತಿಮ ದಿನಾಂಕ 11/26 ಮತ್ತು CVV 456 ನಮೂದಿಸಿ.", "kn", "ಕಾರ್ಡ್‌ನ ಅಂತಿಮ ದಿನಾಂಕ ನವೆಂಬರ್ ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು ಮತ್ತು CVV ನಾಲ್ಕು, ಐದು, ಆರು ನಮೂದಿಸಿ."),
    ("కార్డు గడువు తేదీ 08/25 మరియు CVV 456", "te", "కార్డు గడువు తేదీ ఆగస్టు రెండు వేల ఇరవై ఐదు మరియు CVV నాలుగు, ఐదు, ఆరు"),
    ("एक्सपायरी डेट 12/25 टाका.", "mr", "एक्सपायरी डेट डिसेंबर दोन हजार पंचवीस टाका."),
    # R3 financial-year ranges
    ("ਵਿੱਤੀ ਸਾਲ 2023-24 ਲਈ", "pa", "ਵਿੱਤੀ ਸਾਲ ਦੋ ਹਜ਼ਾਰ ਤੇਈ ਚੌਵੀ ਲਈ"),
    ("वित्त वर्ष 2023-24", "hi", "वित्त वर्ष दो हजार तेईस चौबीस"),
    # R4 space-separated card / phone digit groups
    ("तुमच्या 4111 2233 4455 6677 या कार्डवरून", "mr", "तुमच्या चार, एक, एक, एक, दोन, दोन, तीन, तीन, चार, चार, पाच, पाच, सहा, सहा, सात, सात या कार्डवरून"),
    ("ಒಟಿಪಿಯನ್ನು 98765 43210 ಗೆ ಕಳುಹಿಸುತ್ತಿದ್ದೇನೆ", "kn", "ಒಟಿಪಿಯನ್ನು ಒಂಬತ್ತು, ಎಂಟು, ಏಳು, ಆರು, ಐದು, ನಾಲ್ಕು, ಮೂರು, ಎರಡು, ಒಂದು, ಸೊನ್ನೆಗೆ ಕಳುಹಿಸುತ್ತಿದ್ದೇನೆ"),
    ("call 98765 43210", "en", "call nine, eight, seven, six, five, four, three, two, one, zero"),
    # R5 PIN keyword
    ("कृपया अपना यूपीआई पिन 4532 दर्ज करें", "hi", "कृपया अपना यूपीआई पिन चार, पाँच, तीन, दो दर्ज करें"),
    # R6 six-digit PIN codes
    ("சென்னை 600018 என்ற முகவரியில்", "ta", "சென்னை ஆறு, பூஜ்ஜியம், பூஜ்ஜியம், பூஜ்ஜியம், ஒன்று, எட்டு என்ற முகவரியில்"),
    ("Bengaluru 560038.", "en", "Bengaluru five, six, zero, zero, three, eight."),
    ("नागपूर 440010 येथे", "mr", "नागपूर चार, चार, शून्य, शून्य, एक, शून्य येथे"),
    ("যোৰহাট 785001 ত", "as", "যোৰহাট সাত, আঠ, পাঁচ, শূন্য, শূন্য, এক ত"),
    ("ಪುಣೆಯ 411001 ಅಂಚೆ ಕಚೇರಿಯಲ್ಲಿ", "kn", "ಪುಣೆಯ ನಾಲ್ಕು, ಒಂದು, ಒಂದು, ಸೊನ್ನೆ, ಸೊನ್ನೆ, ಒಂದು ಅಂಚೆ ಕಚೇರಿಯಲ್ಲಿ"),
    # R7 version strings
    ("ऐप को v4.5.2 पर अपडेट करें", "hi", "ऐप को वी चार दशमलव पाँच दशमलव दो पर अपडेट करें"),
    # R8 Tamil Month D, YYYY keeps the year as a year
    ("கடந்த ஆகஸ்ட் 15, 2025 அன்று", "ta", "கடந்த ஆகஸ்ட் பதினைந்தாம் தேதி இரண்டாயிரத்து இருபத்தைந்து அன்று"),
    # R9 rupees + paise in every language
    ("₹1,25,000.50 has been credited", "en", "one lakh twenty five thousand rupees fifty paise has been credited"),
    ("পরিমাণ হল ₹95.41, যা", "bn", "পরিমাণ হল পঁচানব্বই টাকা একচল্লিশ পয়সা, যা"),
    ("મોત્તા (₹5,400.50) નેગેટિવ", "gu", "મોત્તા, પાંચ હજાર ચારસો રૂપિયા પચાસ પૈસા, નેગેટિવ"),
    ("ಮೊತ್ತ ₹4,500.50 ಮರುಪಾವತಿ", "kn", "ಮೊತ್ತ ನಾಲ್ಕು ಸಾವಿರದ ಐನೂರು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ ಮರುಪಾವತಿ"),
    ("ബിൽ ₹95.41 ആണ്", "ml", "ബിൽ തൊണ്ണൂറ്റിയഞ്ച് രൂപ നാൽപ്പത്തിയൊന്ന് പൈസ ആണ്"),
    ("बिल ₹95.41 आहे", "mr", "बिल पंच्याण्णव रुपये एक्केचाळीस पैसे आहे"),
    ("ਬਿੱਲ ₹95.41 ਹੈ", "pa", "ਬਿੱਲ ਪਚਾਨਵੇਂ ਰੁਪਏ ਇਕਤਾਲੀ ਪੈਸੇ ਹੈ"),
    ("தொகை ₹15,430.50 ஆகும்", "ta", "தொகை பதினைந்தாயிரத்து நானூற்று முப்பது ரூபாய் ஐம்பது பைசா ஆகும்"),
])
def test_residual_rules(text, lang, want):
    assert normalise_text(text, lang) == want


@pytest.mark.parametrize("text,lang", [
    ("Your OTP is 482913.", "en"),                       # 6-digit OTP stays digit by digit (already was)
    ("मेरे पास 25 रुपये हैं", "hi"),
])
def test_residual_rules_keep_existing(text, lang):
    from maya3.text.tn import normalize as tn_normalize
    assert normalise_text(text, lang) == tn_normalize(T._caps_fold(text), lang).strip()


@pytest.mark.parametrize("text,lang,want", [
    ("घर नंबर 8-2-45, सेक्टर 17", "hi", None),            # door number: not a date (unchanged from the plain normaliser)
    ("100000 रुपये", "hi", "एक लाख रुपये"),                  # amount: not a PIN code
    ("₹100000 का", "hi", "एक लाख रुपये का"),
    ("customer id CUST-567890", "en", None),               # ID suffix: left to the normaliser's code reader
    ("राशि 150000 ÷ 12 है", "hi", None),                    # arithmetic: not a PIN code
    ("दिनांक 05/11 को", "hi", None),                         # DD/MM without year: unchanged
    ("www.airindia.in/v2.1 देखें", "mr", None),               # inside a URL: rules do not apply
])
def test_residual_rules_negative(text, lang, want):
    from maya3.text.tn import normalize as tn_normalize
    plain = re.sub(r"\s+", " ", tn_normalize(T._caps_fold(text), lang)).strip()
    got = normalise_text(text, lang)
    if want is None:
        assert got == plain.replace(" dash ", " " + T._rules.DASH.get(lang, "dash") + " ")
    else:
        assert got == want
    assert "February" not in got and "फ़रवरी" not in got and "फरवरी" not in got


def test_tag_keeps_comma(monkeypatch):
    # P1 is off by default (no measured gain); when switched on it keeps the comma before an inline [tag]
    assert normalise_text("नौ, आठ, सात, [curious] छह, पाँच", "hi") == "नौ, आठ, सात. [curious] छह, पाँच"
    monkeypatch.setattr(T._rules, "TAG_COMMA", True)
    assert normalise_text("नौ, आठ, सात, [curious] छह, पाँच", "hi") == "नौ, आठ, सात, [curious] छह, पाँच"
    assert normalise_text("आपका पेमेंट फेल हुआ था ना, [sighs] बारह मई को", "hi") == "आपका पेमेंट फेल हुआ था ना, [sighs] बारह मई को"
    assert normalise_text("ठीक है. [laughs] चलो", "hi") == "ठीक है. [laughs] चलो"


def test_uri_dash_native():
    assert normalise_text("kyc_update@safebank-india.com ਤੇ ਭੇਜੋ", "pa") == "ਕੇ ਵਾਈ ਸੀ ਅੰਡਰਸਕੋਰ update ਐਟ ਦ ਰੇਟ safebank ਡੈਸ਼ india ਡਾਟ com ਤੇ ਭੇਜੋ"   # P3 spells kyc
    assert "dash" in normalise_text("mail a-b@x.com", "en")


def test_rules_all_languages():
    for lang in ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"):
        out = normalise_text("15-08-25, 4111 2233 4455 6677, 560038, ₹95.41, v4.5.2, 2023-24, expiry 08/27", lang)
        assert out and not any(ch.isdigit() for ch in out), (lang, out)
        assert "minus" not in out and "/" not in out, (lang, out)


# ---------------------------------------------------------------------------------------------------------------
# normaliser fixes from a multi-language benchmark (maya3.text.lexicon L1-L3, maya3.text.rules P3)
@pytest.mark.parametrize("text,want", [
    ("4 8 9 30 45 71", "চাৰি আঠ ন ত্ৰিশ পঞ্চল্লিশ একসত্তৰ"),
    ("₹1,250.50", "এক হাজাৰ দুশ পঞ্চাশ টকা পঞ্চাশ পইচা"),
    ("2,50,00,000", "দুই কোটি পঞ্চাশ লাখ"),
    ("1st 4th 9th", "প্ৰথম চতুৰ্থ নৱম"),
    ("3.14", "তিনি দশমিক এক চাৰি"),
    ("12/03/2024", "বাৰ মাৰ্চ দুই হাজাৰ চৌবিশ"),
    ("পুৱা 10:30 বজাত বিনিময়", "পুৱা দহ বাজি ত্ৰিশ মিনিটত বিনিময়"),
    ("ৰাতি 8:00 বজাৰ ভিতৰত", "ৰাতি আঠ বজাৰ ভিতৰত"),
    ("12:30 PM ত", "দুপৰীয়া বাৰ বাজি ত্ৰিশ মিনিটত"),
    ("₹50 ৰ এটা কেচবেক", "পঞ্চাশ টকাৰ এটা কেচবেক"),
    ("2.5% লৈকে", "দুই দশমিক পাঁচ শতাংশলৈকে"),
    ("45°C", "পঞ্চল্লিশ ডিগ্ৰী চেলছিয়াছ"),
    ("3/4", "তিনি চতুৰ্থাংশ"),
    ("আমি আজি 2023 টা অভিযোগৰ", "আমি আজি দুই হাজাৰ তেইশ টা অভিযোগৰ"),      # টা is a classifier, not o'clock
    ("support@hdfcbank.com লৈ", "support এট দ্য ৰেট এইচ ডি এফ সি bank ডট com লৈ"),
    ("ID TRN987654", "ID টি, আৰ, এন, ন, আঠ, সাত, ছয়, পাঁচ, চাৰি"),
])
def test_assamese_lexicon(text, want):
    assert normalise_text(text, "as") == want


def test_assamese_no_bengali_forms():
    from maya3.text import lexicon
    T._lexicon.install(True)
    assert all("র" not in w for w in lexicon.AS_ONES) and len(lexicon.AS_ONES) == 100
    s = ("Pay ₹1,23,456.75 on 12/03/2024 at 5:00 PM, 1st, 3rd, 33%, 1/2, x², 5 > 3, a_b@x-y.co.in, #45, A&B, "
         "38°C, RS 42, 9:15 AM, 14:30, 1990 চনত, 99, 31, 38 [laughs]")
    out = normalise_text(s, "as")
    assert "র" not in out and not any(ch.isdigit() for ch in out), out          # no Bengali ra (র), no digits
    for bn_word in ("চার", "আট", "তিন ", "নয়", "সকাল", "দুপুর", "সন্ধ্যা", "অ্যাট", "এর ", "টা বেজে"):
        assert bn_word not in out, (bn_word, out)


def test_lexicon_off_restores_vendored_tables():
    from maya3.text.tn import normalize as tn_normalize
    s = "12:30 PM ত ₹50 ৰ 45, 71 support@hdfcbank.com e-receipt.hdfcbank.com 40 km/h"
    for lang in ("as", "bn", "hi", "kn"):
        on = normalise_text(s, lang)
        off = normalise_text(s, lang, rules=False)
        assert off == re.sub(r"\s+", " ", tn_normalize(T._caps_fold(s), lang)).strip()
        assert on != off
        assert normalise_text(s, lang) == on                       # toggling back on gives the same output
    assert "ত্রিশ" in normalise_text("30", "as", rules=False)       # vendored data untouched
    assert normalise_text("30", "bn") == "ত্রিশ"                     # Bengali keeps its own ra


@pytest.mark.parametrize("text,lang,want", [
    ("40 km/h বেগে", "bn", "চল্লিশ কে এম প্রতি ঘণ্টা বেগে"),
    ("850 km/h আৰু", "as", "আঠশ পঞ্চাশ কে এম প্ৰতি ঘণ্টা আৰু"),
    ("45 km/h च्या", "mr", "पंचेचाळीस के एम प्रति तासच्या"),
    ("120 km/h ની", "gu", "એકસો વીસ કે એમ પ્રતિ કલાકની"),
    ("100 MB/s ಗಿಂತ", "kn", "ನೂರು ಎಂ ಬಿ ಪರ್ ಸೆಕೆಂಡ್ ಗಿಂತ"),
    ("30 km/h வேகத்தில்", "ta", "முப்பது கே எம் பெர் ஹவர் வேகத்தில்"),
    ("speed 40 km/h", "en", "speed forty kilometers per hour"),
    ("100 MB/s", "en", "one hundred M B per second"),
])
def test_per_units(text, lang, want):
    assert normalise_text(text, lang) == want


@pytest.mark.parametrize("text,lang,want", [
    # sub-domain hosts were spelled from the last label only
    ("तुमची पावती e-receipt.hdfcbank.com वर", "mr", "तुमची पावती ई डॅश receipt डॉट एच डी एफ सी bank डॉट com वर"),
    ("netbanking.nationalbank.in ਤੋਂ", "pa", "netbanking ਡਾਟ nationalbank ਡਾਟ in ਤੋਂ"),
    # a bare domain at the end of a sentence was not spelled at all
    ("Book on irctc.co.in.", "en", "Book on I R C T C dot co dot in."),
    # UPI IDs: the local part was left raw
    ("pay rahul.deshmukh@okaxis now", "en", "pay rahul dot deshmukh at okaxis now"),
    # P3 acronym labels
    ("report-fraud@sbi.co.in ত", "as", "report ডেশ্ব fraud এট দ্য ৰেট এছ বি আই ডট co ডট in ত"),
    ("www.onlinesbi.sbi देखें", "hi", "डब्ल्यू डब्ल्यू डब्ल्यू डॉट online एस बी आई डॉट एस बी आई देखें"),
    ("kyc.axisbank.com", "en", "K Y C dot axisbank dot com"),
    ("support@paytm.com", "en", "support at paytm dot com"),                      # not an acronym: unchanged
])
def test_uri_hosts_and_acronyms(text, lang, want):
    assert normalise_text(text, lang) == want


def test_uri_acronym_only_inside_uris():
    assert normalise_text("sbi and kyc", "en") == "sbi and kyc"
    assert normalise_text("Contact the public office", "en") == "Contact the public office"


@pytest.mark.parametrize("text,lang,want", [
    # R10 percent range with two signs ("...percentminus four percent")
    ("3.5%-4% सुत", "as", "তিনি দশমিক পাঁচৰ পৰা চাৰি শতাংশ सुत"),
    ("rate 3.5%-4%", "en", "rate three point five to four percent"),
    # R11 price per unit ("/" dropped, "kg" left raw)
    ("₹500/kg ਲੱਗੇਗਾ", "pa", "ਪੰਜ ਸੌ ਰੁਪਏ ਪ੍ਰਤੀ ਕੇ ਜੀ ਲੱਗੇਗਾ"),
    ("₹500/kg ଅତିରିକ୍ତ", "or", "ପାଞ୍ଚ ଶହ ଟଙ୍କା ପ୍ରତି କେ ଜି ଅତିରିକ୍ତ"),
    ("₹500/kg extra", "en", "five hundred rupees per kg extra"),
    # R12 p.a.
    ("8.5% p.a. सुद", "mr", "आठ पूर्णांक पाच टक्के पी ए सुद"),
    ("7% p.a. interest", "en", "seven percent per annum interest"),
])
def test_symbol_rules(text, lang, want):
    assert normalise_text(text, lang) == want
