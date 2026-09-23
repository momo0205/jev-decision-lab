import tomllib
from pathlib import Path

import yaml


def test_offline_ci_excludes_optional_classifier_tests() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    project = tomllib.loads(Path("pyproject.toml").read_text())
    jobs = workflow["jobs"]
    offline_steps = jobs["offline"]["steps"]
    classifier_steps = jobs["classifier"]["steps"]
    offline_install = next(step["run"] for step in offline_steps if "pip install" in step.get("run", ""))
    classifier_install = next(
        step["run"] for step in classifier_steps if "pip install" in step.get("run", "")
    )
    offline_pytest = next(step["run"] for step in offline_steps if "pytest " in step.get("run", ""))
    classifier_pytest = next(
        step["run"] for step in classifier_steps if "pytest " in step.get("run", "")
    )

    assert "-e '.[dev]'" in offline_install
    assert "classifier" not in offline_install
    assert "-e '.[dev,classifier]'" in classifier_install
    assert "not classifier" in offline_pytest
    assert ".coveragerc.offline" in offline_pytest
    assert ".coveragerc.classifier" in classifier_pytest
    assert "classifier: requires optional classifier dependencies" in {
        marker.split(":", maxsplit=1)[0]: marker
        for marker in project["tool"]["pytest"]["ini_options"]["markers"]
    }.values()
