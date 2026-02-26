from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = REPO_ROOT / 'scripts' / 'runtime_pack_router.py'
CONTRACT_PATH = REPO_ROOT / 'config' / 'runtime_tier_c_pins.json'


class RuntimeTierCContractTest(unittest.TestCase):
    def test_contract_matches_router_pins(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
        contract_profiles = contract.get('profiles')
        self.assertIsInstance(contract_profiles, dict)

        router_profiles = self._extract_router_pins()

        self.assertEqual(
            router_profiles,
            contract_profiles,
            'config/runtime_tier_c_pins.json must match scripts/runtime_pack_router.py::PINNED_TIER_C_PROFILES',
        )

    def _extract_router_pins(self) -> dict[str, int]:
        module = ast.parse(ROUTER_PATH.read_text(encoding='utf-8'), filename=str(ROUTER_PATH))
        for node in module.body:
            value_node = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'PINNED_TIER_C_PROFILES':
                        value_node = node.value
                        break
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == 'PINNED_TIER_C_PROFILES':
                    value_node = node.value

            if value_node is not None:
                parsed = ast.literal_eval(value_node)
                self.assertIsInstance(parsed, dict)
                return {str(k): int(v) for k, v in parsed.items()}

        self.fail('PINNED_TIER_C_PROFILES constant not found in runtime_pack_router.py')


if __name__ == '__main__':
    unittest.main()
