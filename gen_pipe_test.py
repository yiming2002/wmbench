from pipelines.base import WatermarkBasePipeline
from utils.model_loader import ModelLoader
from data_structure.base import WatermarkInput


if __name__ == "__main__":
    model_loader = ModelLoader(
        model_id='Qwen/Qwen2.5-1.5B',
        vocab_size=151936,

        max_new_tokens=200,         # 限制生成长度
        do_sample=True,
        no_repeat_ngram_size=4,
        temperature=0.7,            # 适当的随机性让文本更自然
        repetition_penalty=1.1 
    )

    gen_pipeline = WatermarkBasePipeline(model_loader=model_loader, alg_name="KGW")

    prompt = (
        "李白的诗写得怎么样"
    )
    input = WatermarkInput(prompt=prompt)


    watermarked_text = gen_pipeline.generate_with_watermark(input=input)
    unwatermarked_text = gen_pipeline.generate_without_watermark(input=input)

    result_wm = gen_pipeline.detect(watermarked_text.text)
    result_uw = gen_pipeline.detect(unwatermarked_text.text)

    print("Watermarked Text:\n", watermarked_text.text)
    print("Unwatermarked Text:\n", unwatermarked_text.text)
    print("Watermark Detection Result (Watermarked):", result_wm.is_watermarked)
    print("Watermark Detection Result (Unwatermarked):", result_uw.is_watermarked)