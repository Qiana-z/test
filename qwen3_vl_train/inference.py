# -*- coding: utf-8 -*-
import os, torch
from swift.llm import PtEngine, InferRequest, RequestConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'
os.environ['SWIFT_DEBUG'] = '0'


engine = PtEngine("model/Qwen/Qwen3-Omni-30B-A3B-Instruct", attn_impl='sdpa')  # "sdpa", "flash_attention_2"

# infer_request = InferRequest(messages=[{
#     "role": "user",
#     "content": '<video>介绍这个视频。',
# }], videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'])

infer_request = InferRequest(messages=[{
    "role": "user",
    "content": [
        {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
        {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
    ]
}])

request_config = RequestConfig(max_tokens=128, temperature=0)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

print("Before infer:")
print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("reserved :", torch.cuda.memory_reserved() / 1024**3, "GB")

resp_list = engine.infer([infer_request], request_config=request_config)

torch.cuda.synchronize()
print("After infer:")
print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("reserved :", torch.cuda.memory_reserved() / 1024**3, "GB")
print("peak allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")

response = resp_list # [0].choices[0] # .message.content  # should return: 'A baby wearing glasses sits on a bed, engrossed in reading a book. The baby turns the pages with both hands, occasionally looking up and smiling. The room is cozy, with a crib in the background and clothes scattered around. The baby’s focus and curiosity are evident as they explore the book, creating a heartwarming scene of early learning and discovery.'
print(response)

# use stream
# request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
# gen_list = engine.infer([infer_request], request_config=request_config)
# for chunk in gen_list[0]:
#     if chunk is None:
#         continue
#     print(chunk.choices[0].delta.content, end='', flush=True)
# print()