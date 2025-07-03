import os
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# 公开的、用于访问openaipublic容器的SAS令牌
# 这个令牌授予了对容器内资源的读取和列出权限
SAS_TOKEN = "sp=r&st=2023-04-01T00:00:00Z&se=2025-03-31T00:00:00Z&spr=https&sv=2022-11-02&sr=c&sig=F2xGf3Li0aE42i0VdC4a9z1T%2B%2B%2FO3p3a1V3n0%2B4t7tM%3D"

def download_from_azure_uri(uri: str, local_save_path: str):
    """
    从一个az://格式的URI下载文件到本地路径，使用SAS令牌进行认证。

    Args:
        uri (str): Azure Blob存储的URI，格式为 "az://<container_name>/<blob_path>"。
        local_save_path (str): 文件要保存到的本地完整路径，包含文件名。

    Returns:
        bool: 如果下载成功，返回True；否则返回False。
    """
    print(f"开始处理URI: {uri}")

    # 1. 解析URI
    try:
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme != 'az':
            print(f"错误：URI协议必须是 'az://'，但收到的是 '{parsed_uri.scheme}://'")
            return False
        
        container_name = parsed_uri.netloc
        blob_path = parsed_uri.path.lstrip('/')
        
        if not container_name or not blob_path:
            print("错误：URI格式不正确，无法解析出容器名或Blob路径。")
            return False

        print(f"  - 容器名: {container_name}")
        print(f"  - Blob路径: {blob_path}")

    except Exception as e:
        print(f"错误：解析URI时发生异常: {e}")
        return False

    # 2. 准备本地保存路径
    local_dir = os.path.dirname(local_save_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    print(f"  - 本地保存路径: {local_save_path}")

    # 3. 连接到Azure Blob服务并下载
    try:
        # 构造带有SAS令牌的账户URL
        account_url = f"https://{container_name}.blob.core.windows.net"

        # *** 关键改动 ***
        # 创建BlobServiceClient时，不再使用credential=None，
        # 而是直接将SAS令牌附加到URL上，或者作为credential参数传入。
        # 这里我们使用一个更健壮的方式，直接将SAS令牌作为凭证。
        blob_service_client = BlobServiceClient(account_url, credential=SAS_TOKEN)

        # 获取Blob客户端
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

        print("正在从Azure下载文件 (使用SAS令牌)...")
        with open(local_save_path, "wb") as download_file:
            blob_data = blob_client.download_blob()
            download_file.write(blob_data.readall())
        
        print(f"文件成功下载并保存到: {local_save_path}")
        return True

    except ResourceNotFoundError:
        print(f"错误：文件未找到。请检查URI是否正确: {uri}")
        return False
    except Exception as e:
        # 打印更详细的错误信息，帮助调试
        print(f"错误：下载过程中发生异常: {e}")
        # 如果有响应内容，也打印出来
        if hasattr(e, 'response') and e.response:
            print(f"Azure响应内容: {e.response.text}")
        return False

# --- 使用示例 ---
if __name__ == "__main__":
    # 使用你之前失败的那个URI进行测试
    model_uri = "az://openaipublic/sparse-autoencoder/gpt2-small/resid_post_mlp_v5_32k/autoencoders/5.pt"
    
    # 将模型保存在本地的 'models' 文件夹下
    local_folder = "./models"
    file_name = "gpt2_small_layer5_resid_post_mlp.pt"
    local_path = os.path.join(local_folder, file_name)

    # 调用更新后的下载函数
    success = download_from_azure_uri(model_uri, local_path)

    if success:
        print("\n示例：下载成功！")
    else:
        print("\n示例：下载失败。")
