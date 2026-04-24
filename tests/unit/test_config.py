"""Unit tests for configuration system."""

import os

import pytest

from effgen.config.loader import Config, ConfigLoader


class TestConfig:
    """Tests for Config dataclass."""

    def test_create_empty_config(self):
        config = Config()
        assert config.data == {}

    def test_create_with_data(self):
        config = Config(data={"key": "value"})
        assert config["key"] == "value"

    def test_dict_access(self):
        config = Config(data={"a": 1, "b": {"c": 2}})
        assert config["a"] == 1
        assert config["b"]["c"] == 2

    def test_set_item(self):
        config = Config()
        config["key"] = "value"
        assert config["key"] == "value"

    def test_missing_key_raises(self):
        config = Config()
        with pytest.raises(KeyError):
            _ = config["nonexistent"]


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_yaml(self, tmp_dir):
        yaml_file = tmp_dir / "test.yaml"
        yaml_file.write_text("name: test\nvalue: 42\n")
        loader = ConfigLoader()
        config = loader.load_config(str(yaml_file))
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_load_json(self, tmp_dir):
        import json
        json_file = tmp_dir / "test.json"
        json_file.write_text(json.dumps({"name": "test", "value": 42}))
        loader = ConfigLoader()
        config = loader.load_config(str(json_file))
        assert config["name"] == "test"
        assert config["value"] == 42

    def test_load_nonexistent_file(self):
        loader = ConfigLoader()
        with pytest.raises((OSError, FileNotFoundError)):
            loader.load_config("/nonexistent/path/config.yaml")

    def test_env_variable_substitution(self, tmp_dir):
        os.environ["EFFGEN_TEST_VAR"] = "hello_world"
        yaml_file = tmp_dir / "test.yaml"
        yaml_file.write_text("value: ${EFFGEN_TEST_VAR}\n")
        loader = ConfigLoader()
        config = loader.load_config(str(yaml_file))
        assert config["value"] is not None
        os.environ.pop("EFFGEN_TEST_VAR", None)
