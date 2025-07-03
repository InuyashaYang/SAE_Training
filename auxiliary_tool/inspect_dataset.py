import h5py

def print_h5_structure(name, obj):
    """递归打印 H5 文件结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"数据集 '{name}': 形状 {obj.shape}, 类型 {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"组 '{name}':")

print("检查H5文件结构...")
with h5py.File("/data0/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/test.h5", "r") as f:
    # 使用 visititems 递归打印所有内容
    print("\nH5文件结构:")
    f.visititems(print_h5_structure)
    
    # 直接打印第一层 keys
    print("\n第一层结构:")
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"数据集 '{key}': 形状 {f[key].shape}, 类型 {f[key].dtype}")
        else:
            print(f"组 '{key}'")
    
    print("\n其他属性:")
    for attr in f.attrs:
        print(f"属性 '{attr}': {f.attrs[attr]}")
