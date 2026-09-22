"""Packaging boundary for the reusable RelEnt semantic kernel."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


class RelEntPortabilityBoundaryTests(unittest.TestCase):
    def test_relent_never_imports_parliament_host_modules(self):
        root = Path(__file__).resolve().parents[2]
        violations: list[str] = []
        for path in sorted((root / "relent").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = str(node.module or "")
                    if module == "global_workspace" or module.startswith(
                        "global_workspace."
                    ):
                        violations.append(f"{path.relative_to(root)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "global_workspace" or alias.name.startswith(
                            "global_workspace."
                        ):
                            violations.append(
                                f"{path.relative_to(root)}:{node.lineno}"
                            )
        self.assertEqual(
            violations,
            [],
            "RelEnt must remain host-neutral; move Parliament integration to "
            "global_workspace/relent_adapt.py or world_admission.py",
        )

    def test_production_uses_world_admission_facade(self):
        root = Path(__file__).resolve().parents[2]
        forbidden = {
            "parse_world_model",
            "world_model_from_dict",
            "compile_chance_gated_world",
            "validate_world_model",
            "validate_world_completeness",
            "admit_world_model_extension",
        }
        allowed = {"world_state.py", "world_admission.py"}
        violations: list[str] = []
        for path in sorted((root / "global_workspace").glob("*.py")):
            if path.name in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = str(node.module or "")
                if module != "world_state" or node.level != 1:
                    continue
                imported = forbidden & {alias.name for alias in node.names}
                if imported:
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}:"
                        + ",".join(sorted(imported))
                    )
        self.assertEqual(
            violations,
            [],
            "Production admission must enter through global_workspace.world_admission",
        )


if __name__ == "__main__":
    unittest.main()
