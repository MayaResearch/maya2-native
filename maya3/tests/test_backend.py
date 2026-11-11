from types import SimpleNamespace

from serve.backend import detect_backend


class FakeCuda:
    def __init__(self, available=True, metadata_error=False):
        self.available = available
        self.metadata_error = metadata_error

    def is_available(self):
        return self.available

    def current_device(self):
        if self.metadata_error:
            raise RuntimeError("driver metadata unavailable")
        return 0

    def get_device_name(self, index):
        if self.metadata_error:
            raise RuntimeError("driver metadata unavailable")
        return "Test GPU"


def fake_torch(*, hip=None, cuda=None, available=True, metadata_error=False):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=FakeCuda(available, metadata_error),
        __version__="test-version",
    )


def test_rocm_is_not_misidentified_as_nvidia_cuda():
    info = detect_backend(fake_torch(hip="7.1.25424", cuda=None), probe_triton=False)
    assert info.name == "rocm"
    assert info.available
    assert info.hip_version == "7.1.25424"
    assert info.cuda_version is None
    assert info.device_name == "Test GPU"


def test_nvidia_cuda_detection():
    info = detect_backend(fake_torch(cuda="12.8"), probe_triton=False)
    assert info.name == "cuda"
    assert info.available
    assert info.cuda_version == "12.8"
    assert info.hip_version is None


def test_cuda_build_identity_is_preserved_without_available_gpu():
    info = detect_backend(fake_torch(cuda="12.8", available=False), probe_triton=False)
    assert info.name == "cuda"
    assert not info.available
    assert info.cuda_version == "12.8"
    assert info.device_name is None
    assert info.device_index is None


def test_cpu_detection():
    info = detect_backend(fake_torch(available=False), probe_triton=False)
    assert info.name == "cpu"
    assert not info.available
    assert info.device_name is None


def test_device_metadata_failure_does_not_break_backend_detection():
    info = detect_backend(
        fake_torch(cuda="12.8", available=True, metadata_error=True),
        probe_triton=False,
    )
    assert info.name == "cuda"
    assert info.available
    assert info.device_name is None
    assert info.device_index is None
