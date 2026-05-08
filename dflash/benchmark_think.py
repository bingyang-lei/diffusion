import argparse
import time
import random
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def find_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    limit = len(sequence) - len(pattern) + 1
    for i in range(limit):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def split_acceptance_lengths_by_phase(
    step_token_counts: list[int],
    before_token_count: int,
) -> tuple[list[int], list[int]]:
    # Acceptance length is a per-step metric, so each decode step should be
    # assigned to exactly one phase instead of being split across both sides
    # of </think>.
    before = []
    after = []
    cursor = 0

    for step_tokens in step_token_counts:
        if step_tokens <= 0:
            continue

        step_start = cursor
        if step_start < before_token_count:
            before.append(step_tokens)
        else:
            after.append(step_tokens)
        cursor += step_tokens

    return before, after

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    ) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    decode_step_token_counts = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        accepted_tokens = acceptance_length + 1
        acceptance_lengths.append(accepted_tokens)
        decode_step_token_counts.append(accepted_tokens)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    has_stop_token = False
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            has_stop_token = True
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens
    is_truncated = (not has_stop_token) and (num_output_tokens >= max_new_tokens)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_step_token_counts=decode_step_token_counts,
        is_truncated=is_truncated,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking in the prompt (off by default)")
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip baseline evaluation for block-size=1 and related outputs.",
    )
    args = parser.parse_args()
    print("now the dataset we evaluate is: ", args.dataset)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    print(f"Using block size: {block_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            bs_list = [block_size] if args.skip_base else list(dict.fromkeys([1, block_size]))
            for bs in bs_list:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

            spec_response.generated_token_ids = generated_ids.tolist()
            
            # Avoid sending large GPU tensors through gather_object.
            for bs in response:
                response[bs].output_ids = None

            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    mean_num_output_tokens = np.mean([r[block_size].num_output_tokens for r in responses])
    max_num_output_tokens = np.max([r[block_size].num_output_tokens for r in responses])
    print(f"dflash max num output tokens: {max_num_output_tokens}")
    print(f"dflash mean num output tokens: {mean_num_output_tokens}")
    print(f"dflash time per output token: {tb}")
    print(f"dflash mean throughput: {1 / tb:.2f} tokens/s")

    if not args.skip_base:
        t1 = np.mean([r[1].time_per_output_token for r in responses])
        baseline_mean_num_output_tokens = np.mean([r[1].num_output_tokens for r in responses])
        max_baseline_num_output_tokens = np.max([r[1].num_output_tokens for r in responses])
        print(f"baseline max num output tokens: {max_baseline_num_output_tokens}")
        print(f"baseline mean num output tokens: {baseline_mean_num_output_tokens}")
        print(f"baseline time per output token: {t1}")
        print(f"baseline mean throughput: {1 / t1:.2f} tokens/s")
        print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if args.enable_thinking:
        before_token_counts = []
        after_token_counts = []

        before_phase_acceptance = []
        after_phase_acceptance = []
        excluded_truncated_thinking_samples = 0

        for r in responses:
            result = r[block_size]
            token_ids = result.generated_token_ids
            close_start_idx = find_subsequence(token_ids, think_close_ids)
            close_found = close_start_idx >= 0
            close_end_idx = close_start_idx + len(think_close_ids) - 1 if close_found else -1
            before_count = close_end_idx + 1 if close_found else len(token_ids)
            after_count = len(token_ids) - before_count

            before_token_counts.append(before_count)
            after_token_counts.append(after_count)

            # If generation is truncated while still thinking, keep its length stats,
            # but exclude it from phase-specific acceptance metrics.
            exclude_acceptance = (not close_found) and result.is_truncated
            if exclude_acceptance:
                excluded_truncated_thinking_samples += 1
                continue

            before_steps, after_steps = split_acceptance_lengths_by_phase(
                step_token_counts=result.decode_step_token_counts,
                before_token_count=before_count,
            )
            if len(before_steps) > 0:
                before_phase_acceptance.append(np.mean(before_steps))
            if len(after_steps) > 0:
                after_phase_acceptance.append(np.mean(after_steps))

        mean_before_tokens = np.mean(before_token_counts) if len(before_token_counts) > 0 else 0.0
        mean_after_tokens = np.mean(after_token_counts) if len(after_token_counts) > 0 else 0.0
        print(f"Mean generated tokens before </think>: {mean_before_tokens:.2f}")
        print(f"Mean generated tokens after </think>: {mean_after_tokens:.2f}")

        if len(before_phase_acceptance) > 0:
            print(f"Average Acceptance length before </think>: {np.mean(before_phase_acceptance):.2f}")
        else:
            print("Average Acceptance length before </think>: N/A")

        if len(after_phase_acceptance) > 0:
            print(f"Average Acceptance length after </think>: {np.mean(after_phase_acceptance):.2f}")
        else:
            print("Average Acceptance length after </think>: N/A")

        print(
            "Excluded truncated always-thinking samples from split acceptance stats: "
            f"{excluded_truncated_thinking_samples}"
        )

if __name__ == "__main__":
    main()