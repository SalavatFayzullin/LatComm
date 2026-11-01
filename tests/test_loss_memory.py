"""Verify the memory-saving CE path against ordinary CE, including gradients."""
import pytest
import torch

from latcomm.evaluate import score_responses


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_checkpointed_loss_preserves_values_gradients_and_option_scores(dtype):
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(2, 77, 29, generator=generator, dtype=dtype)
    batch = {"response_ids": torch.randint(29, (2, 77), generator=generator),
             "response_mask": torch.ones(2, 77, dtype=torch.bool),
             "option_ids": torch.tensor([1, 3, 5]),
             "option_mask": torch.tensor([[True, True, False], [True, True, True]])}
    batch["response_mask"][0, 60:] = False
    weights = torch.rand(2, 77, generator=generator)
    plain = source.clone().requires_grad_(True)
    recomputed = source.clone().requires_grad_(True)
    expected = score_responses(plain, batch)
    actual = score_responses(recomputed, batch, checkpoint_loss=True)
    assert torch.equal(actual.nll, expected.nll)
    assert torch.equal(actual.predictions, expected.predictions)
    assert torch.equal(actual.correct_tokens, expected.correct_tokens)
    (expected.nll * weights).sum().backward()
    (actual.nll * weights).sum().backward()
    torch.testing.assert_close(recomputed.grad, plain.grad, rtol=0, atol=0)


def test_checkpointed_loss_does_not_retain_all_fp32_vocabulary_buffers():
    batch = {"response_ids": torch.zeros(1, 96, dtype=torch.long),
             "response_mask": torch.ones(1, 96, dtype=torch.bool)}

    def retained_float32_bytes(enabled):
        storage_sizes = {}
        def pack(tensor):
            if tensor.dtype == torch.float32 and tensor.numel():
                storage = tensor.untyped_storage()
                storage_sizes[storage.data_ptr()] = storage.nbytes()
            return tensor
        logits = torch.randn(1, 96, 1024, dtype=torch.float16, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            score = score_responses(logits, batch, checkpoint_loss=enabled)
        assert score.nll.requires_grad
        return sum(storage_sizes.values())

    assert retained_float32_bytes(False) >= 96 * 1024 * 4
    assert retained_float32_bytes(True) == 0
