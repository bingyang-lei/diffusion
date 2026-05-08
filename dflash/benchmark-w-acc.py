import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import argparse
import ast
import time
import random
import importlib.util
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
from datasets import load_dataset

_KL_FILE_PATH = _THIS_DIR / "benchmark-variant" / "benchmark_kl_divergence.py"
_KL_SPEC = importlib.util.spec_from_file_location("benchmark_kl_divergence", _KL_FILE_PATH)
if _KL_SPEC is None or _KL_SPEC.loader is None:
    raise ImportError(f"Cannot load module from {_KL_FILE_PATH}")
_KL_MODULE = importlib.util.module_from_spec(_KL_SPEC)
_KL_SPEC.loader.exec_module(_KL_MODULE)
extract_model_answer = _KL_MODULE.extract_model_answer
is_aime_correct = _KL_MODULE.is_aime_correct
is_math500_correct = _KL_MODULE.is_math500_correct

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        LatexNormalizationConfig,
        parse,
        verify,
    )
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def is_math500_correct_with_math_verify(pred_text: str, gold_text: str) -> bool:
    if not HAS_MATH_VERIFY:
        return is_math500_correct(pred_text, gold_text)
    try:
        gold_parsed = parse(gold_text)
        pred_parsed = parse(
            pred_text,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    normalization_config=LatexNormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=False,
                        nits=False,
                        boxed="all",
                        equations=False,
                    ),
                ),
                ExprExtractionConfig(),
            ],
        )
        return verify(gold_parsed, pred_parsed)
    except Exception:
        # Fallback to previous logic if parsing/verification fails unexpectedly.
        return is_math500_correct(pred_text, gold_text)

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def parse_show_id(show_id_raw: str | None) -> set[int] | None:
    if show_id_raw is None:
        return None
    text = show_id_raw.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, int):
            return {parsed}
        if isinstance(parsed, (list, tuple, set)):
            ids = {int(x) for x in parsed}
            return {x for x in ids if x >= 0}
    except Exception:
        pass

    # Fallback for comma-separated forms like "1,2,3,4".
    items = [x.strip() for x in text.strip("[](){}").split(",") if x.strip()]
    ids = set()
    for item in items:
        try:
            value = int(item)
            if value >= 0:
                ids.add(value)
        except ValueError:
            continue
    return ids if ids else None


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

        acceptance_lengths.append(acceptance_length+1)
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
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
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
    parser.add_argument(
        "--output-content",
        action="store_true",
        help="Print question text and full model response content.",
    )
    parser.add_argument(
        "--show-id",
        type=str,
        default=None,
        help="Filter printed samples by 1-based ids, e.g. \"[1,2,3]\" or \"1,2,3\". "
        "Only used when --output-content is enabled.",
    )
    args = parser.parse_args()
    show_id_set = parse_show_id(args.show_id)

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
    answer_list = []
    if args.dataset in ("aime24", "aime25"):
        if args.dataset == "aime24":
            dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        else:
            dataset = load_dataset("math-ai/aime25", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        answer_list = [item.get("answer", "") for item in dataset]
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = (
            "{problem}\n"
            "Please reason step by step.\n"
            "Your final answer must be exactly one LaTeX expression inside \\boxed{{}} "
            "(for example: \\boxed{{\\frac{{1}}{{2}}}}), and place it on the last line.\n"
            "Use standard LaTeX math formatting in your response: write fractions as "
            "\\frac{{a}}{{b}}, text labels as \\text{{...}}, and degree symbols as ^\\circ "
            "(for example, 90^\\circ)."
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        answer_list = [item.get("answer", "") for item in dataset]
    else:
        dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
    if args.dataset in ("aime24", "aime25", "math500"):
        answer_list = [item.get("answer", "") for item in dataset]

    responses = []
    if args.output_content and show_id_set is not None:
        selected_indices = sorted(i for i in show_id_set if 0 <= i < len(dataset))
        indices = selected_indices[dist.rank() :: dist.size()]
    else:
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

            predictions = {}
            for bs in bs_list:
                gen = response[bs]
                generated_ids = gen.output_ids[0, gen.num_input_tokens:]
                predictions[bs] = tokenizer.decode(generated_ids, skip_special_tokens=True)

            
            # Avoid sending large GPU tensors through gather_object.
            for bs in response:
                response[bs].output_ids = None

            responses.append(
                {
                    "idx": idx,
                    "metrics": response,
                    "predictions": predictions,
                    "turns": instance.get("turns", []),
                }
            )

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    responses.sort(key=lambda x: x["idx"])
    response_metrics = [item["metrics"] for item in responses]

    if args.output_content:
        # Content-inspection mode: print requested cases only.
        printed = 0
        for item in responses:
            dataset_idx = int(item.get("idx"))
            if show_id_set is not None and dataset_idx not in show_id_set:
                continue
            turns = item.get("turns", [])
            preds = item.get("predictions", {})
            print("-" * 80)
            print(f"sample_id: {dataset_idx}")
            if turns:
                print("question:")
                for t_idx, turn_text in enumerate(turns, start=1):
                    print(f"[Q{t_idx}] {turn_text}")
            else:
                print("question: (missing turns)")
            print(f"dflash_response (bs={block_size}):")
            print(preds.get(block_size, ""))
            if not args.skip_base and 1 in preds:
                print("base_response (bs=1):")
                print(preds.get(1, ""))
            printed += 1
        if printed == 0:
            print("No matching sample_id found for current dataset split.")
        return

    tb = np.mean([r[block_size].time_per_output_token for r in response_metrics])
    mean_num_output_tokens = np.mean([r[block_size].num_output_tokens for r in response_metrics])
    max_num_output_tokens = np.max([r[block_size].num_output_tokens for r in response_metrics])
    print(f"dflash max num output tokens: {max_num_output_tokens}")
    print(f"dflash mean num output tokens: {mean_num_output_tokens}")
    print(f"dflash time per output token: {tb}")
    print(f"dflash mean throughput: {1 / tb:.2f} tokens/s")

    if not args.skip_base:
        t1 = np.mean([r[1].time_per_output_token for r in response_metrics])
        baseline_mean_num_output_tokens = np.mean([r[1].num_output_tokens for r in response_metrics])
        max_baseline_num_output_tokens = np.max([r[1].num_output_tokens for r in response_metrics])
        print(f"baseline max num output tokens: {max_baseline_num_output_tokens}")
        print(f"baseline mean num output tokens: {baseline_mean_num_output_tokens}")
        print(f"baseline time per output token: {t1}")
        print(f"baseline mean throughput: {1 / t1:.2f} tokens/s")
        print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in response_metrics])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in response_metrics]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if args.dataset in ("aime24", "aime25", "math500") and answer_list:
        num_eval = min(len(answer_list), len(responses))
        correct_dflash = 0
        correct_base = 0

        def check_correct(pred_text: str, gold_text: str) -> bool:
            pred_answer = extract_model_answer(pred_text)
            if args.dataset in ("aime24", "aime25"):
                return is_aime_correct(pred_answer, gold_text)
            if args.dataset == "math500":
                # For MATH-500, evaluate against the full model output with math-verify.
                return is_math500_correct_with_math_verify(pred_text, gold_text)
            return is_math500_correct(pred_answer, gold_text)

        for i in range(num_eval):
            gold_answer = str(answer_list[i])
            preds = responses[i].get("predictions", {})

            dflash_pred = preds[block_size]
            correct_dflash += int(check_correct(dflash_pred, gold_answer))

            if not args.skip_base:
                base_pred = preds.get(1)
                if base_pred is not None:
                    correct_base += int(check_correct(base_pred, gold_answer))

        dflash_acc = correct_dflash / num_eval if num_eval > 0 else 0.0
        print(
            f"{args.dataset} dflash accuracy: "
            f"{dflash_acc * 100:.2f}% ({correct_dflash}/{num_eval})"
        )

        if not args.skip_base:
            base_acc = correct_base / num_eval if num_eval > 0 else 0.0
            print(
                f"{args.dataset} base accuracy: "
                f"{base_acc * 100:.2f}% ({correct_base}/{num_eval})"
            )

if __name__ == "__main__":
    main()