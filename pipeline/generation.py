from typing import List, Optional, Tuple

import torch
from transformers import LogitsProcessor


def generate_with_watermark(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    processor: Optional[LogitsProcessor] = None,
    temperature: float = 0.75,
) -> Tuple[str, List[int]]:
    """Generate text with optional watermark logits processor."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = input_ids

        # Prime KV cache once with prompt, then decode incrementally.
        outputs = model(input_ids=generated, use_cache=True)
        past_key_values = outputs.past_key_values

        for _ in range(max_tokens):
            logits = outputs.logits[:, -1, :] / temperature

            if processor is not None:
                logits = processor(generated, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

    generated_ids = generated[0].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids
