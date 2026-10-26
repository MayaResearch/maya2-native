"""Maya3 inference: fast streaming engine, voice registry with cached prefix KV, engine-internal guards, public API."""
__all__ = ["Maya3TTS", "Engine", "EngineConfig", "Voice"]


def __getattr__(name):
    if name == "Maya3TTS":
        from maya3.api import Maya3TTS
        return Maya3TTS
    if name in ("Engine", "EngineConfig"):
        from maya3 import engine
        return getattr(engine, name)
    if name == "Voice":
        from maya3.prompt import Voice
        return Voice
    raise AttributeError(name)
