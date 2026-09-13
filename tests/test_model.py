import unittest
from dataclasses import replace

import torch

from ced.config import ModelConfig
from ced.generate import generate
from ced.model import CEDLanguageModel


class CEDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = ModelConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=64,
            window_size=4,
            max_seq_len=64,
        )
        self.model = CEDLanguageModel(self.config).eval()

    def test_future_tokens_cannot_change_prefix_logits(self) -> None:
        tokens = torch.randint(0, 256, (2, 19))
        original = self.model(tokens).logits
        for boundary in (1, 4, 11):
            changed = tokens.clone()
            changed[:, boundary:] = (changed[:, boundary:] + 17) % 256
            actual = self.model(changed).logits
            torch.testing.assert_close(
                actual[:, :boundary], original[:, :boundary], atol=1e-7, rtol=1e-6
            )

    def test_future_input_embeddings_have_zero_gradient(self) -> None:
        model = CEDLanguageModel(replace(self.config, tie_embeddings=False))
        tokens = torch.arange(12).unsqueeze(0)
        model(tokens).logits[:, 5].square().sum().backward()
        gradient = model.embedding.weight.grad
        self.assertGreater(gradient[:6].abs().sum().item(), 0)
        self.assertEqual(gradient[6:12].abs().sum().item(), 0)

    def test_exact_cache_matches_full_forward_across_windows_and_kv_groups(self) -> None:
        tokens = torch.randint(0, 256, (2, 19))
        for groups in (1, 2):
            model = CEDLanguageModel(replace(self.config, decoder_kv_groups=groups)).eval()
            full = model(tokens).logits
            for prefix in (1, 4, 9):
                with self.subTest(groups=groups, prefix=prefix):
                    result = model.prefill(tokens[:, :prefix])
                    torch.testing.assert_close(
                        result.logits, full[:, prefix - 1 : prefix], atol=2e-6, rtol=1e-5
                    )
                    position = prefix
                    for chunk in (1, 6, 20):
                        end = min(position + chunk, tokens.size(1))
                        if end == position:
                            continue
                        result = model.decode(tokens[:, position:end], result.cache)
                        torch.testing.assert_close(
                            result.logits, full[:, position:end], atol=2e-6, rtol=1e-5
                        )
                        self.assertEqual(result.cache.length, end)
                        self.assertEqual(len(result.cache.memory), groups)
                        position = end

    def test_global_decoder_kv_depends_only_on_encoder(self) -> None:
        tokens = torch.randint(0, 256, (2, 13))
        before = self.model.prefill(tokens)
        with torch.no_grad():
            for block in self.model.decoder:
                block.ffn.down.weight.normal_(std=0.5)
                block.attention.local_kv.linear.weight.normal_(std=0.5)
        after = self.model.prefill(tokens)
        for first, second in zip(before.cache.memory, after.cache.memory, strict=True):
            torch.testing.assert_close(first.key, second.key, atol=0, rtol=0)
            torch.testing.assert_close(first.value, second.value, atol=0, rtol=0)
        self.assertFalse(torch.allclose(before.logits, after.logits))
        self.assertTrue(all(layer.global_kv is None for layer in after.cache.decoder))

    def test_bounded_replay_processes_only_window_and_is_marked_approximate(self) -> None:
        tokens = torch.randint(0, 256, (2, 17))
        lengths = []
        hooks = [
            block.register_forward_pre_hook(lambda module, args: lengths.append(args[0].size(1)))
            for block in self.model.decoder
        ]
        bounded = self.model.prefill(tokens, mode="bounded")
        for hook in hooks:
            hook.remove()
        self.assertEqual(lengths, [4, 4])
        self.assertTrue(bounded.cache.approximate)
        self.assertEqual(bounded.cache.decoder_prefill_tokens, 4)
        self.assertEqual(bounded.cache.memory[0].end, 17)
        exact = self.model.prefill(tokens)
        self.assertGreater((exact.logits - bounded.logits).abs().max().item(), 1e-9)
        continued = self.model.decode(torch.tensor([[1], [2]]), bounded.cache)
        self.assertTrue(continued.cache.approximate)
        self.assertTrue(torch.isfinite(continued.logits).all())

    def test_short_bounded_prefill_is_exact(self) -> None:
        tokens = torch.randint(0, 256, (1, 4))
        full = self.model.prefill(tokens)
        bounded = self.model.prefill(tokens, mode="bounded")
        self.assertFalse(bounded.cache.approximate)
        torch.testing.assert_close(full.logits, bounded.logits, atol=0, rtol=0)

    def test_local_caches_have_bounded_allocations(self) -> None:
        for batch_size in (1, 2):
            tokens = torch.randint(0, 256, (batch_size, 29))
            result = self.model.prefill(tokens)
            for layer in (*result.cache.encoder, *result.cache.decoder):
                self.assertEqual(layer.local.start, 25)
                for tensor in (layer.local.key, layer.local.value):
                    self.assertEqual(tensor.size(2), 4)
                    self.assertEqual(
                        tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size()
                    )

    def test_requests_can_interleave_without_hidden_cache_state(self) -> None:
        first, second = torch.randint(256, (1, 12)), torch.randint(256, (1, 7))
        a = self.model.prefill(first[:, :5])
        self.model.prefill(second)
        actual = self.model.decode(first[:, 5:], a.cache)
        expected = self.model(first).logits[:, 5:]
        torch.testing.assert_close(actual.logits, expected, atol=2e-6, rtol=1e-5)

    def test_gradients_reach_both_stacks_and_global_kv_projection(self) -> None:
        self.model.train()
        sequence = torch.randint(256, (2, 17))
        result = self.model(sequence[:, :-1], sequence[:, 1:])
        result.loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_model_can_overfit_a_small_sequence(self) -> None:
        self.model.train()
        sequence = torch.tensor(([10, 20, 30, 40, 50] * 5)[:25]).repeat(4, 1)
        x, y = sequence[:, :-1], sequence[:, 1:]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = self.model(x, y).loss.item()
        for _ in range(45):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model(x, y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        final = self.model(x, y).loss.item()
        self.assertLess(final, initial * 0.15)

    def test_greedy_cached_generation_matches_full_recomputation(self) -> None:
        prompt = torch.tensor([[256, 72, 105]])
        actual = generate(self.model, prompt, max_new_tokens=8, temperature=0)
        expected = prompt
        with torch.no_grad():
            for _ in range(8):
                logits = self.model(expected).logits[:, -1].clone()
                logits[:, 256] = -torch.inf
                token = logits.argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
                if token.item() == 257:
                    break
        torch.testing.assert_close(actual, expected)

    def test_rejects_context_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.model(torch.ones(1, 65, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            generate(self.model, torch.ones(1, 60, dtype=torch.long), max_new_tokens=8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_mixed_precision_forward_backward_and_cache(self) -> None:
        model = self.model.cuda().train()
        tokens = torch.randint(256, (2, 17), device="cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            loss = model(tokens[:, :-1], tokens[:, 1:]).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.autocast("cuda", dtype=dtype):
            full = model(tokens).logits
            result = model.prefill(tokens[:, :9])
            result = model.decode(tokens[:, 9:], result.cache)
        torch.testing.assert_close(result.logits, full[:, 9:], atol=0.01, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
