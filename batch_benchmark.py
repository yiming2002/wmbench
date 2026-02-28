import argparse
import time

from data_structure.base import WatermarkInput
from pipelines.base import WatermarkBasePipeline
from utils.model_loader import ModelLoader


def build_prompts(num_prompts: int) -> list[WatermarkInput]:
    return [
        WatermarkInput(prompt=f"请用三句话介绍李白的诗歌风格，并给出一个例子。样本编号：{idx}")
        for idx in range(num_prompts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num-prompts", type=int, default=48)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--detect-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_loader = ModelLoader(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        no_repeat_ngram_size=4,
        temperature=0.7,
        repetition_penalty=1.1,
    )

    pipeline = WatermarkBasePipeline(model_loader=model_loader, alg_name="KGW")
    inputs = build_prompts(args.num_prompts)

    start = time.perf_counter()
    wm_outputs = pipeline.generate_with_watermark_batch(inputs, batch_size=args.gen_batch_size)
    wm_time = time.perf_counter() - start

    start = time.perf_counter()
    uw_outputs = pipeline.generate_without_watermark_batch(inputs, batch_size=args.gen_batch_size)
    uw_time = time.perf_counter() - start

    start = time.perf_counter()
    wm_results = pipeline.detect_batch([item.text for item in wm_outputs], batch_size=args.detect_batch_size)
    uw_results = pipeline.detect_batch([item.text for item in uw_outputs], batch_size=args.detect_batch_size)
    detect_time = time.perf_counter() - start

    wm_positive = sum(1 for item in wm_results if item.is_watermarked)
    uw_positive = sum(1 for item in uw_results if item.is_watermarked)

    # print(f"prompts={args.num_prompts}")
    print(f"wm_generate_time={wm_time:.2f}s throughput={args.num_prompts / wm_time:.2f} prompts/s")
    print(f"uw_generate_time={uw_time:.2f}s throughput={args.num_prompts / uw_time:.2f} prompts/s")
    print(f"detect_time={detect_time:.2f}s throughput={(2 * args.num_prompts) / detect_time:.2f} texts/s")
    # print(f"wm_detect_positive={wm_positive}/{len(wm_results)}")
    # print(f"uw_detect_positive={uw_positive}/{len(uw_results)}")


if __name__ == "__main__":
    main()
