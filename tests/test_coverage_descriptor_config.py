import unittest

from coverage.descriptor_config import SoapDescriptorConfig


def _valid_kwargs(**overrides):
    kwargs = dict(
        r_cut=2.0, n_max=2, l_max=1, sigma=0.3, species=("H",), rbf="gto", periodic=True,
    )
    kwargs.update(overrides)
    return kwargs


class SoapDescriptorConfigTests(unittest.TestCase):
    def test_construction_with_all_required_fields_succeeds(self):
        config = SoapDescriptorConfig(**_valid_kwargs())
        self.assertEqual(config.r_cut, 2.0)
        self.assertEqual(config.species, ("H",))

    def test_missing_field_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            SoapDescriptorConfig(**{k: v for k, v in _valid_kwargs().items() if k != "sigma"})
        self.assertIn("sigma", str(ctx.exception))

    def test_no_field_has_an_implicit_default(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig()

    def test_empty_species_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(species=()))

    def test_non_tuple_species_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(species=["H"]))

    def test_blank_species_entry_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(species=("H", "  ")))

    def test_non_positive_r_cut_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(r_cut=0))

    def test_non_positive_n_max_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(n_max=0))

    def test_negative_l_max_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(l_max=-1))

    def test_non_positive_sigma_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(sigma=0))

    def test_invalid_rbf_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(rbf="bogus"))

    def test_non_bool_periodic_rejected(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(periodic=1))

    def test_bool_rejected_for_numeric_fields(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig(**_valid_kwargs(n_max=True))

    def test_content_hash_is_deterministic(self):
        a = SoapDescriptorConfig(**_valid_kwargs())
        b = SoapDescriptorConfig(**_valid_kwargs())
        self.assertEqual(a.content_hash(), b.content_hash())

    def test_content_hash_changes_with_any_scientific_parameter(self):
        base = SoapDescriptorConfig(**_valid_kwargs())
        changed = SoapDescriptorConfig(**_valid_kwargs(r_cut=2.5))
        self.assertNotEqual(base.content_hash(), changed.content_hash())

    def test_from_dict_round_trip(self):
        payload = {
            "r_cut": 2.0, "n_max": 2, "l_max": 1, "sigma": 0.3,
            "species": ["H", "O"], "rbf": "gto", "periodic": True,
        }
        config = SoapDescriptorConfig.from_dict(payload)
        self.assertEqual(config.species, ("H", "O"))

    def test_from_dict_rejects_unknown_field(self):
        payload = dict(_valid_kwargs())
        payload["aggregation_policy"] = "max"
        with self.assertRaises(ValueError) as ctx:
            SoapDescriptorConfig.from_dict(payload)
        self.assertIn("aggregation_policy", str(ctx.exception))

    def test_from_dict_rejects_non_dict_payload(self):
        with self.assertRaises(ValueError):
            SoapDescriptorConfig.from_dict(["not", "a", "dict"])


if __name__ == "__main__":
    unittest.main()
