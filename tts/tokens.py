"""Token contract v0 for the custom-TTS campaign (see custom-tts-plan.md §5).

The backbone keeps its text tokenizer untouched; we append ~50 control tokens to the
text vocab and keep audio OUTSIDE the text vocab entirely (audio enters as embeddings
summed from the depth codebooks, never as text-vocab ids).

Sequence layout of one utterance (sentence-interleave format):
    <|lang:xx|> text ... <|emo|> emo_label <|audio|> a a a ... <aeos-as-cb0-class>
The emotion slot sits AFTER the text and BEFORE the audio so the model predicts it
causally from text (and forced values steer the audio that follows). Pace tags are
plain-text tags inside the text span; punctuation is the prosody interface.

<aeos> is NOT a token in this table: it is class N_CODES (2048) on the cb0 head only,
weighted 3x in the loss. Keeping it out of the text vocab means the backbone can never
emit it as text.
"""
from dataclasses import dataclass, field

# ---- audio codec constants (Kyutai Mimi, 32-codebook lane) -------------------------
FRAME_HZ = 12.5
N_Q = 32                 # codebooks trained; serve-time early-exit at 16/24 allowed
N_CODES = 2048           # per-codebook vocabulary
AEOS_CLASS = N_CODES     # cb0 head has N_CODES+1 classes; class 2048 = <aeos>
AEOS_LOSS_WEIGHT = 3.0

LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

CTL_SLOTS = 32           # reserved, unassigned control ids for future use


@dataclass(frozen=True)
class TokenContract:
    """Names of every added text-vocab token, in the exact order they are appended.

    The order is FROZEN once a checkpoint exists: appending is allowed, reordering
    and deleting are not. `added_tokens()` is the single source of truth consumed by
    the tokenizer extension, the embedding-resize code and the golden contract test.
    """
    spk: str = "<|spk|>"                  # precedes the speaker-embedding soft slot
    audio: str = "<|audio|>"              # text->audio switch
    emo: str = "<|emo|>"                  # emotion slot opener (label follows as text)
    instruct_open: str = "<|instruct|>"   # instruction-mode span (Mode-B controls)
    instruct_close: str = "<|/instruct|>"
    cont: str = "<|cont|>"                # non-initial continuation span (from
                                          # tag_continuation; never a mere packing
                                          # boundary)
    langs: tuple = tuple(f"<|lang:{l}|>" for l in LANGS)
    ctl: tuple = tuple(f"<|ctl:{i:02d}|>" for i in range(CTL_SLOTS))

    def added_tokens(self) -> list:
        return [self.spk, self.audio, self.emo,
                self.instruct_open, self.instruct_close, self.cont,
                *self.langs, *self.ctl]

    # next-token loss-mask policy: these are conditioning
    # inputs, never prediction targets; <|emo|> and <|audio|> stay unmasked
    # (learned causal sentinels), as does an inferred emotion label.
    def masked_as_targets(self) -> list:
        return [self.spk, self.instruct_open, self.instruct_close, self.cont,
                *self.langs, *self.ctl]


CONTRACT = TokenContract()


def extend_tokenizer(tokenizer):
    """Append the contract tokens; return (tokenizer, name->id dict).

    Idempotent: HF add_tokens skips names already present, and ids are re-read from
    the tokenizer, so calling this on an already-extended checkpoint is a no-op.
    """
    tokenizer.add_tokens(CONTRACT.added_tokens(), special_tokens=True)
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in CONTRACT.added_tokens()}
    missing = [t for t, i in ids.items() if i is None or i < 0]
    if missing:
        raise ValueError(f"tokenizer failed to register: {missing}")
    return tokenizer, ids


def golden_contract(tokenizer) -> dict:
    """Serializable snapshot for the golden tensor-contract test."""
    _, ids = extend_tokenizer(tokenizer)
    return {
        "frame_hz": FRAME_HZ, "n_q": N_Q, "n_codes": N_CODES,
        "aeos_class": AEOS_CLASS, "aeos_loss_weight": AEOS_LOSS_WEIGHT,
        "added": ids, "vocab_size": len(tokenizer),
    }
