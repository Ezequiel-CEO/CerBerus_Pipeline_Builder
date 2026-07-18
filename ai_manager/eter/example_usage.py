from .llama_cpp_loader import LlamaCppLoader
from cerberus_api.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Configurações do modelo
    model_path = "/caminho/para/seu/modelo.gguf"
    config = {
        "n_gpu_layers": -1,  # Todas as camadas na GPU
        "n_ctx": 2048,  # Tamanho do contexto
        "n_batch": 512,  # Tamanho do batch
        "verbose": False,  # Modo verboso
    }

    # Inicializa o loader
    loader = LlamaCppLoader(model_path, **config)

    try:
        # Carrega o modelo
        loader.load()

        # Exemplo de prompt
        prompt = "Explique o que é inteligência artificial em português."

        # Configurações de geração
        gen_config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
        }

        # Gera texto
        response = loader.generate(prompt, **gen_config)
        logger.info(f"Resposta: {response}")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
    finally:
        # Descarrega o modelo
        loader.unload()


if __name__ == "__main__":
    main()
