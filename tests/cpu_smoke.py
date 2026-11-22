"""Check pinned TRL/PEFT APIs on a tiny random model, NOT the real experiment."""
import math
import tempfile
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

vocab = {"<pad>":0, "<eos>":1, "<unk>":2, "request":3, "card":4, "cash":5}
raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
raw.pre_tokenizer = WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
config = LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=64, eos_token_id=1, pad_token_id=0)
data = Dataset.from_list([{"prompt":"request ", "completion":"card<eos>"},
                         {"prompt":"request ", "completion":"cash<eos>"}])
with tempfile.TemporaryDirectory() as out:
    args = SFTConfig(output_dir=out, max_steps=1, per_device_train_batch_size=1,
                     max_length=32, completion_only_loss=True, packing=False,
                     use_cpu=True, bf16=False, fp16=False, gradient_checkpointing=False,
                     save_strategy="no", report_to="none")
    trainer = SFTTrainer(model=LlamaForCausalLM(config), processing_class=tokenizer,
                         args=args, train_dataset=data,
                         peft_config=LoraConfig(r=2, lora_alpha=4,
                             target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM"))
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    trainer.save_model(out + "/adapter")
    reloaded = PeftModel.from_pretrained(LlamaForCausalLM(config), out + "/adapter")
    assert reloaded.peft_config
print("Synthetic CPU integration check passed; no SmolLM3 quality result.")
