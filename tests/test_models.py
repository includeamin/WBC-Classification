import pytest
import torch

from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import set_backbone_frozen
from wbc_classification.models.registry import MODEL_NAMES, build_model, supports_freezing


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    model = build_model(name, num_classes=4).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 4)


def test_baseline_accepts_tiny_and_odd_sizes():
    model = build_model("baseline", num_classes=4).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 16, 16)).shape == (1, 4)
        assert model(torch.zeros(1, 3, 50, 70)).shape == (1, 4)


def test_unknown_model_lists_choices():
    with pytest.raises(ConfigError, match="resnet18"):
        build_model("vgg", num_classes=4)


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
def test_freeze_leaves_only_head_trainable(name):
    model = build_model(name, num_classes=4)
    set_backbone_frozen(model, True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(("fc", "classifier")) for n in trainable)
    set_backbone_frozen(model, False)
    assert all(p.requires_grad for p in model.parameters())


def test_supports_freezing():
    assert supports_freezing("resnet18") and not supports_freezing("baseline")


def test_pretrained_download_failure_is_a_config_error(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("wbc_classification.models.backbone.tv_models.get_model", offline)
    with pytest.raises(ConfigError, match="resnet18"):
        build_model("resnet18", num_classes=4, pretrained=True)
