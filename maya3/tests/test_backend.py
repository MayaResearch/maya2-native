from types import SimpleNamespace

from serve.backend import detect_backend


class FakeCuda:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return "Test GPU"


def fake_torch(*, hip=None, cuda=None, available=True):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=FakeCuda(available),
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
    assert info.cuda_version == "12.8"
    assert info.hip_version is None


def test_cpu_detection():
    info = detect_backend(fake_torch(available=False), probe_triton=False)
    assert info.name == "cpu"
    assert not info.available
    assert info.device_name is None
