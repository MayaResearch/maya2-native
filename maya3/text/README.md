# maya3.text: the bundled text normaliser

`normalise_text(text, lang)` turns numbers, currency, dates, times, OTPs and phone numbers, URLs and e-mails into
spoken words in the request language (as bn en gu hi kn ml mr or pa ta te). It title-cases ALL-CAPS names and leaves
inline `[tags]` alone. The server applies it to every request by default. `--no-normalise` turns it off for the
whole server, and the request field `"normalize": false` turns it off for one request.
`Maya3TTS.generate/stream(..., normalize=True)` applies it in Python.

**Provenance.** `tn/` is vendored byte for byte from Maya Research's text normaliser (upstream commit `fe5e6ec`); the release benchmark used the same copy. `tests/test_text.py::test_vendored_copy_is_verbatim` pins the sha256 of each file (one comment line in `indic_languages.py` was reworded for the public release). The only addition is `tn/__init__.py`. The wrapper in `__init__.py` adds caps folding, tag protection and a per-piece fallback: on error it returns the piece unchanged.

**Dependencies.** Standard library only. The package has no data files.

**Not supported.** The opt-in identifier modes `FR3_TN_URL_SPELL=1` and `FR3_TN_LONG_DIGITS=1` read
`identifier_words.json`. That file is not in the fe5e6ec snapshot. With either variable set, the normaliser raises on any text
that contains a URL, e-mail address or long digit run. The wrapper then passes that piece (the text between
`[tags]`) through unnormalised. Both are off by default, as they were in the evals. When the normaliser
falls back internally (an invalid date, for example), it prints one `[obs] {"event": "normalization_fallback", ...}`
line to stdout. That line never contains the caller's text.

**Targeted rules (`rules.py`).** `normalise_text(text, lang)` applies a small set of pre/post rules around
the vendored copy (which stays byte-identical). They come from a review of the remaining failures in the release evaluations. Each one fixes an input shape that the strict normaliser rejected (it then read
the span character by character) or read in an unexpected way:

| rule | input | before | after |
|---|---|---|---|
| R1 date_2y | `15-08-25`, `9/9/99` | "one five minus zero eight minus two five" | "fifteen August two thousand twenty five" (00-50 -> 20xx) |
| R2 expiry | `11/26` beside an expiry/validity/date word or a CVV | "eleven by twenty six" | "November two thousand twenty six" |
| R3 fy_range | `2023-24` | "two, zero, two, three, two, four" | "two thousand twenty three twenty four" |
| R4 digit_groups | `4111 2233 4455 6677`, `98765 43210` | four cardinals | digit by digit |
| R5 pin_keyword | `पिन 4532` | "four thousand five hundred thirty two" | digit by digit |
| R6 pincode | bare 6-digit number, no currency/unit/ID prefix/maths next to it | "five lakh sixty thousand thirty eight" | digit by digit |
| R7 version | `v4.5.2` | "four.five.two" | "four point five point two" |
| R8 ta_date_year | Tamil `ஆகஸ்ட் 15, 2025` | year read digit by digit | year read as a year |
| R9 paise | `₹95.41` in bn en gu kn ml mr pa ta | "ninety five point four one rupees" | "ninety five rupees forty one paise" |
| P1 tag_comma | `सात, [curious] छह` | "सात. [curious]" | the comma is kept (off by default: no measured gain) |
| P2 uri_dash | `-` inside an e-mail or URL, Indic languages | English "dash" | the word in the request language |

Rules never apply inside e-mail addresses or URLs. `normalise_text(..., rules=False)` reproduces the eval-time
behaviour exactly (`test_parity_with_eval_wrapper`). That mode is for testing only, because R9 patches the
module-level language tables while rules are on. The before/after diff over all 13,468 eval texts (517 changed) was
reviewed by hand.

**Benchmark review fixes.** More corrections, again around the
untouched vendored copy and only with `rules=True`:

| fix | where | before | after |
|---|---|---|---|
| L1 as_lexicon | `lexicon.py` | Assamese reused Bengali tables: ত্রিশ (Bengali র), প্রথম/চতুর্থ, অ্যাট দ্য রেট, letters আর/এস, সকাল/দুপুর/রাত, "{h}টা", "দহ বাজি ত্রিশ মিনিট বজাত", "পঞ্চাশ টকা ৰ" | ত্ৰিশ, প্ৰথম/চতুৰ্থ, এট দ্য ৰেট, আৰ/এছ, পুৱা/দুপৰীয়া/ৰাতি, "{h} বজা", "দহ বাজি ত্ৰিশ মিনিটত", "পঞ্চাশ টকাৰ"; 45 পঞ্চল্লিশ, 71 একসত্তৰ |
| L2 per_unit | `lexicon.py` | `40 km/h` -> "কে এম per এইচ" | "কে এম প্রতি ঘণ্টা" (language's per word + hour/second) |
| L3 uri_hosts | `lexicon.py` | `e-receipt.hdfcbank.com` -> "e-receipt.hdfcbank ডট com"; sentence-final `irctc.co.in.` and UPI IDs `a.b@okaxis` left raw | whole host / ID spelled |
| P3 uri_acronym | `rules.py` | `sbi.co.in`, `hdfcbank.com` -> "sbi", "hdfcbank" (Latin) | letters in the request language: "এছ বি আই", "এইচ ডি এফ সি bank" |
| R10 pct_range | `rules.py` | `3.5%-4%` -> "...percentminus four percent" | "three point five to four percent" |
| R11 per_price | `rules.py` | `₹500/kg` -> "five hundred rupees kg" | "... per K G" (per word of the language) |
| R12 per_annum | `rules.py` | `p.a.` left raw | "P A" in the request language (en: "per annum") |

`lexicon.install(False)` restores every patched table, so `rules=False` is still the eval-time behaviour.
