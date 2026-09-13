import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from ced.config import ModelConfig
from ced.data import TokenCorpus
from ced.prepare import prepare
from ced.runtime import Checkpoint, load_checkpoint, resolve_device, save_checkpoint
from ced.tokenizer import ByteTokenizer
from ced.train import TrainOptions, run_training


class DataAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_resolve_device_resolves_auto_and_explicit_index(self) -> None:
        self.assertIn(resolve_device("auto").type, ("cpu", "cuda"))
        self.assertEqual(resolve_device("cpu").type, "cpu")
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda:0").type, "cuda")

    def test_utf8_round_trip_including_special_characters(self) -> None:
        tokenizer = ByteTokenizer()
        text = "你好，CED！🙂\nCafé\x00"
        encoded = tokenizer.encode(text, bos=True, eos=True)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertTrue(all(0 <= token < 258 for token in encoded))

    def test_data_batches_are_shifted_and_file_corruption_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            prepare(["abcdefghijklmnopqrstuvwxyz" * 10], directory)
            corpus = TokenCorpus(directory)
            corpus.check_length(16)
            x, y = corpus.batch("train", 4, 16, torch.Generator().manual_seed(3), "cpu")
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            self.assertEqual(x.dtype, torch.long)
            data_path = directory / "train.bin"
            corrupted = bytearray(data_path.read_bytes())
            corrupted[0] ^= 1
            data_path.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "does not match"):
                TokenCorpus(directory)

    def test_resume_reproduces_uninterrupted_training_including_dropout_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(
                [f"document {i}: the cat sat on the mat. " * 5 for i in range(20)], root / "data"
            )
            config = replace(
                ModelConfig(),
                d_model=32,
                ffn_dim=64,
                window_size=4,
                dropout=0.1,
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            common = [
                "--config",
                str(config_path),
                "--data",
                str(root / "data"),
                "--steps",
                "4",
                "--batch-size",
                "2",
                "--seq-len",
                "16",
                "--grad-accum",
                "2",
                "--eval-interval",
                "2",
                "--eval-batches",
                "1",
                "--log-interval",
                "2",
                "--device",
                "cpu",
                "--warmup-steps",
                "1",
                "--threads",
                "2",
            ]
            middle = root / "middle.pt"

            def also_save_middle(path: Path, payload: Checkpoint) -> None:
                save_checkpoint(path, payload)
                if path.name == "last.pt" and payload["step"] == 2:
                    save_checkpoint(middle, payload)

            with patch("ced.train.save_checkpoint", side_effect=also_save_middle):
                run_training(TrainOptions.from_cli(common + ["--out", str(root / "full")]))
            run_training(
                TrainOptions.from_cli(
                    common + ["--out", str(root / "resumed"), "--resume", str(middle)],
                )
            )
            full = load_checkpoint(root / "full" / "last.pt")
            resumed = load_checkpoint(root / "resumed" / "last.pt")
            self.assertEqual(resumed["step"], 4)
            self.assertEqual(full["val_loss"], resumed["val_loss"])
            for name, tensor in full["model"].items():
                torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
            for param_id, state in full["optimizer"]["state"].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(
                        tensor,
                        resumed["optimizer"]["state"][param_id][name],
                        rtol=0,
                        atol=0,
                    )
            torch.testing.assert_close(
                full["rng"]["sampler"], resumed["rng"]["sampler"], atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
