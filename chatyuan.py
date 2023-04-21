from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda")
tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
model = (
    T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")
    .half()
    .to(device)
)

import clueai

api_key=""
cl = clueai.Client(api_key)

#精调代码 需要训练时解除注释

engine_key = None
'''
response = cl.upload_finetune_corpus(
      file_path="qa_test.json",
      input_field="question",
      target_field="answer",
      base_model_name="ChatYuan"
      )

if "engine_key" in response:
  engine_key = response["engine_key"]
  print("engine key: ", engine_key)
else:
  print(response)
  
'''
#模型精调后，使用engine key进行部署



response = cl.start_finetune_model(
        engine_key=engine_key,
        base_model_name="ChatYuan")
print(response)



def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace("%20", "  ")


def answer(text, context=""):
    """sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样"""
    text = f"{context}\n用户：{text}\n小元："
    text = text.strip()
    text = preprocess(text)
    
    #调参
    generate_config = {
    "do_sample": True,
    "top_p": 0.8,
    "max_length": 128,
    "min_length": 10,
    "length_penalty": 1.0,
    "num_beams": 1
  }
 
 #模型训练后
    if not engine_key:
        prediction = cl.finetune_generate(
        engine_key=engine_key,
        prompt=text,
        base_model_name="ChatYuan")
#未训练
    else:
        prediction = cl.generate(
            model_name='ChatYuan-large',
            prompt=text)
    prediction_text = prediction.generations[0].text
    return postprocess(prediction_text)


class ChatYuan:
    def __init__(self) -> None:
        pass

    def text2text(input_text):
        output_text = answer(input_text)
        return output_text


if __name__ == "__main__":
    text2text = ChatYuan.text2text("时间是否存在")
    print(text2text)
