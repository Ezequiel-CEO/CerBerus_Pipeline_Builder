from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import torch


@dataclass
class QwenConfig:
    """Configuração específica para Qwen2.5-7B-Instruct"""

    model_path: str = "Qwen2.5-7B-Instruct"
    max_length: int = 256  # Reduzido para economizar memória
    temperature: float = 0.8  # Ajustado para seu pedido
    top_p: float = 0.8  # Valor original do modelo
    top_k: int = 20  # Valor original do modelo
    repetition_penalty: float = 1.05  # Valor original do modelo
    do_sample: bool = True  # Valor original do modelo
    max_context_length: int = 5
    use_4bit: bool = False  # Trocado para False para usar 8-bit
    use_8bit: bool = True  # Usando 8-bit que é mais estável
    trust_remote_code: bool = True
    device_map: str = "auto"  # Deixar o framework decidir a melhor distribuição
    torch_dtype: str = "float16"  # Mudar para float16 para compatibilidade

    # Configurações específicas
    use_flash_attention: bool = False  # Desligando para economizar memória
    use_cache: bool = True
    use_safetensors: bool = True

    # Configuração de CPU offloading
    offload_folder: str = "offload_folder"
    cpu_offload: bool = True

    # Configurações de memória
    max_memory: Optional[Dict[int, str]] = None

    def __post_init__(self):
        """Configurações pós-inicialização"""
        # Configurar memória máxima por GPU - extremamente conservador
        if torch.cuda.is_available():
            # Limitar GPU a apenas 1GB
            total_gpu_ram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_limit = min(
                1.0, total_gpu_ram * 0.25
            )  # 25% da GPU ou 1GB, o que for menor

            self.max_memory = {0: f"{gpu_limit:.1f}GB", "cpu": "32GB"}

        # Ajustar caminho do modelo
        self.model_path = str(Path(self.model_path).absolute())
