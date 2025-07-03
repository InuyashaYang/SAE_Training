from transformer_lens import HookedTransformer

# 加载一个模型，例如 gpt2-small
model = HookedTransformer.from_pretrained("gpt2-small")

# 打印出所有可用的钩子点名称
print(f"模型 '{model.cfg.model_name}' 中所有可用的钩子点：")
all_hook_names = [hook.name for hook in model.hook_points]
for name in all_hook_names:
    print(name)

print(f"\n总共有 {len(all_hook_names)} 个钩子点。")
