import os
import sys
import time
import numpy as np
import threading
import ctypes
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
import cupy as cp


class EterCUDAAccelerator:
    """
    Sistema de aceleração que integra kernels CUDA nativos otimizados
    ao framework CerBerus, substituindo implementações Python/CuPy por
    versões native C++/CUDA para operações críticas.
    """

    def __init__(self, engine=None, cuda_path=None, verbose=False):
        """
        Inicializa o acelerador CUDA para o Éter

        Args:
            engine: instância do CerBerusEngine para ser usado como fallback
            cuda_path: caminho para a biblioteca CUDA (opcional)
            verbose: se True, ativa logs detalhados
        """
        # Inicializa logger
        self.logger = logging.getLogger("EterCUDAAccelerator")
        self.verbose = verbose

        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Flags de inicialização e recursos
        self.is_initialized = False
        self.has_tensor_cores = False
        self.has_fp16_support = False
        self.lib = None

        # Caminho base para os kernels CUDA nativos
        self.cuda_src_path = Path("/home/agressor/CerBerusFMK/CerBerusFMK/core_native/src/cuda/kernels")
        self._setup_cuda_paths(cuda_path)

        # Mapeamento de operações para kernels otimizados disponíveis
        self.available_kernels = self._discover_kernels()

        # Estatísticas de aceleração
        self.acceleration_stats = {}

        # Kernels já carregados para uso imediato
        self.loaded_kernels = {}

        # Cache para resultados de benchmarks
        self.benchmark_results = {}

        # Engine para fallback
        self.engine = engine

        # Tenta inicializar bibliotecas nativas
        try:
            self._init_native_libraries()
            self._register_native_functions()
            self._detect_gpu_capabilities()
        except Exception as e:
            self.logger.warning(f"Falha ao inicializar acelerador CUDA: {e}")
            self.is_initialized = False

        # Marca como inicializado
        self.is_initialized = len(self.available_kernels) > 0 or self.lib is not None

        self.logger.info(
            f"Éter CUDA Accelerator inicializado - {len(self.available_kernels)} kernels nativos encontrados"
        )
        self.logger.info(
            f"Status de recursos: Tensor Cores: {self.has_tensor_cores}, FP16: {self.has_fp16_support}"
        )

    def _setup_cuda_paths(self, cuda_path):
        """Configura os caminhos para os arquivos CUDA necessários"""
        # Tenta encontrar o diretório src/cuda
        cuda_paths = [
            Path("/home/agressor/CerBerusFMK/src/cuda"),
            Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            / "core_native"
            / "src"
            / "cuda",
            Path(os.getcwd()) / "src" / "cuda",
            Path(os.getcwd()) / "core_native" / "src" / "cuda",
        ]

        for path in cuda_paths:
            if path.exists():
                self.cuda_src_path = path
                self.logger.info(f"Diretório CUDA encontrado em: {path}")
                break

    def _init_native_libraries(self):
        """Inicializa bibliotecas nativas do Éter CUDA"""
        try:
            # Biblioteca Kennedy integrada (já tem tudo que precisamos)
            lib_paths = [
                "/home/agressor/Documentos/CerBerus_Qantum/kennedy_sync_integrated.so",
                # Caminhos antigos como fallback
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../core_native/build/libcerberus_cuda.so",
                ),
                "/usr/local/lib/libcerberus_cuda.so",
            ]

            self.lib = None
            for path in lib_paths:
                if os.path.exists(path):
                    self.logger.info(f"Carregando biblioteca nativa: {path}")
                    self.lib = ctypes.CDLL(path)
                    break

            if self.lib is None:
                self.logger.warning(
                    "Biblioteca nativa CUDA não encontrada. Usando fallback para CuPy."
                )
                return

            # Registra funções disponíveis
            self._register_native_functions()

        except Exception as e:
            self.logger.error(f"Erro ao inicializar bibliotecas nativas: {str(e)}")
            self.lib = None

    def _register_native_functions(self):
        """Registra funções nativas disponíveis na biblioteca C++/CUDA"""
        if self.lib is None:
            return

        try:
            # Funções de inicialização/detecção
            self._register_function("initialize_cuda", [ctypes.c_int], ctypes.c_bool)
            self._register_function("has_tensor_cores", [], ctypes.c_bool)
            self._register_function("has_fp16_support", [], ctypes.c_bool)

            # Funções para operações matemáticas
            self._register_function(
                "matmul_cuda",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_bool,
                ],
                ctypes.c_bool,
            )

            self._register_function(
                "layer_norm_cuda",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                ],
                ctypes.c_bool,
            )

            self._register_function(
                "gelu_cuda",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                ctypes.c_bool,
            )

            self._register_function(
                "softmax_cuda",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_bool,
            )

            # Inicializa CUDA
            if hasattr(self.lib, "initialize_cuda"):
                device_id = 0  # Usa o primeiro dispositivo por padrão
                if self.lib.initialize_cuda(device_id):
                    self.logger.info("CUDA inicializado com sucesso")
                else:
                    self.logger.warning(
                        "Falha ao inicializar CUDA na biblioteca nativa"
                    )

        except Exception as e:
            self.logger.error(f"Erro ao registrar funções nativas: {str(e)}")

    def _register_function(self, name, argtypes, restype):
        """Registra uma função da biblioteca nativa com os tipos de argumentos e retorno corretos"""
        if not hasattr(self.lib, name):
            self.logger.warning(f"Função '{name}' não encontrada na biblioteca nativa")
            return

        func = getattr(self.lib, name)
        func.argtypes = argtypes
        func.restype = restype

        self.logger.debug(f"Função '{name}' registrada com sucesso")

    def _detect_gpu_capabilities(self):
        """Detecta os recursos disponíveis na GPU"""
        try:
            # Verifica disponibilidade de Tensor Cores
            self.logger.debug("Verificando disponibilidade de Tensor Cores...")
            if self.lib and hasattr(self.lib, "check_tensor_cores"):
                self.has_tensor_cores = bool(self.lib.check_tensor_cores())
                self.logger.debug(f"Tensor Cores disponíveis: {self.has_tensor_cores}")
            else:
                self.has_tensor_cores = False
                self.logger.debug(
                    "Função de verificação de Tensor Cores não disponível"
                )

            # Verifica suporte a FP16
            self.logger.debug("Verificando suporte a FP16...")
            if self.lib and hasattr(self.lib, "check_fp16_support"):
                self.has_fp16_support = bool(self.lib.check_fp16_support())
                self.logger.debug(f"Suporte a FP16 disponível: {self.has_fp16_support}")
            else:
                self.has_fp16_support = False
                self.logger.debug(
                    "Função de verificação de suporte a FP16 não disponível"
                )
        except Exception as e:
            self.logger.warning(f"Erro ao detectar capacidades da GPU: {e}")
            self.has_tensor_cores = False
            self.has_fp16_support = False

    def _discover_kernels(self) -> Dict[str, str]:
        """Descobre kernels CUDA disponíveis no sistema"""
        kernels = {}

        # Procura por kernels na pasta src/cuda
        if self.cuda_src_path.exists():
            # Kernels para operações matemáticas básicas
            if (self.cuda_src_path / "kernels_optimized.cu").exists():
                kernels["matmul"] = str(self.cuda_src_path / "kernels_optimized.cu")
                kernels["layernorm"] = str(self.cuda_src_path / "kernels_optimized.cu")
                kernels["gelu"] = str(self.cuda_src_path / "kernels_optimized.cu")
                kernels["softmax"] = str(self.cuda_src_path / "kernels_optimized.cu")

            # Kernels específicos
            for kernel_file in self.cuda_src_path.glob("*.cu"):
                name = kernel_file.stem
                if "matmul" in name.lower():
                    kernels["matmul_specialized"] = str(kernel_file)
                elif "attention" in name.lower():
                    kernels["attention"] = str(kernel_file)
                elif "fused" in name.lower():
                    kernels["fused_ops"] = str(kernel_file)
                elif "quant" in name.lower():
                    kernels["quantization"] = str(kernel_file)

        return kernels

    def accelerate_operations(self):
        """Ativa aceleração CUDA nativa para todas operações suportadas"""
        operations = list(self.available_kernels.keys())

        self.logger.info(
            f"Éter: Ativando aceleração CUDA nativa para {len(operations)} operações"
        )

        for op in operations:
            # Faz benchmark antes de ativar para comparação
            baseline_perf = self._benchmark_current_implementation(op)

            # Ativa aceleração para esta operação
            success = self._activate_kernel(op)

            if success:
                # Verifica ganho de performance
                accelerated_perf = self._benchmark_current_implementation(op)
                speedup = (
                    baseline_perf / accelerated_perf if accelerated_perf > 0 else 0
                )

                self.acceleration_stats[op] = {
                    "baseline_ms": baseline_perf * 1000,  # em ms
                    "accelerated_ms": accelerated_perf * 1000,  # em ms
                    "speedup": speedup,
                    "status": "active",
                }

                self.logger.info(
                    f"Éter: Aceleração ativada para {op} - Speedup de {speedup:.2f}x"
                )
            else:
                self.logger.warning(f"Éter: Falha ao ativar aceleração para {op}")

    def _activate_kernel(self, operation: str) -> bool:
        """Ativa kernel CUDA nativo para uma operação específica"""
        if operation not in self.available_kernels:
            return False

        kernel_path = self.available_kernels[operation]

        try:
            # Lê o código CUDA do kernel
            with open(kernel_path, "r") as f:
                kernel_code = f.read()

            # Analisa o código para identificar funções e assinaturas
            kernel_functions = self._extract_kernel_functions(kernel_code, operation)

            if not kernel_functions:
                self.logger.warning(
                    f"Nenhuma função de kernel identificada para {operation}"
                )
                return False

            # Injetar codigo que carrega o kernel nativo no engine
            self._inject_kernel_loader(operation, kernel_functions)

            # Cache o kernel para uso
            self.loaded_kernels[operation] = kernel_functions

            return True

        except Exception as e:
            self.logger.error(f"Erro ao ativar kernel para {operation}: {str(e)}")
            return False

    def _extract_kernel_functions(self, code: str, operation: str) -> List[Dict]:
        """Extrai funções de kernel e suas assinaturas do código CUDA"""
        functions = []

        # Procura por assinaturas de funções em CUDA
        # Exemplo: __global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K)

        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "__global__" in line and "void" in line and "(" in line:
                name = line.split("void ")[1].split("(")[0].strip()

                if self._is_related_to_operation(name, operation):
                    # Extrai parâmetros da função
                    params_str = line.split("(")[1]
                    if ")" in params_str:
                        params_str = params_str.split(")")[0]

                    # Extrai corpo da função
                    body_start = i + 1
                    body_end = body_start
                    bracket_count = 1

                    for j in range(body_start, len(lines)):
                        if "{" in lines[j]:
                            bracket_count += lines[j].count("{")
                        if "}" in lines[j]:
                            bracket_count -= lines[j].count("}")

                        if bracket_count == 0:
                            body_end = j
                            break

                    functions.append(
                        {
                            "name": name,
                            "params": params_str,
                            "body_start": body_start,
                            "body_end": body_end,
                            "line": i,
                        }
                    )

        return functions

    def _is_related_to_operation(self, kernel_name: str, operation: str) -> bool:
        """Verifica se o nome do kernel está relacionado à operação"""
        kernel_name = kernel_name.lower()
        operation = operation.lower()

        # Mapeamento de operações para possíveis nomes de kernel
        op_map = {
            "matmul": ["matmul", "gemm", "matrix_mul", "sgemm", "matrix_multiply"],
            "layernorm": ["layernorm", "layer_norm", "norm", "normalize"],
            "gelu": ["gelu", "gaussian", "activation"],
            "softmax": ["softmax", "soft_max", "attention"],
            "attention": ["attention", "self_attention", "mha", "multihead"],
            "fused_ops": ["fused", "fusion", "combined"],
            "quantization": ["quant", "int8", "int4", "dequant"],
        }

        if operation in op_map:
            for pattern in op_map[operation]:
                if pattern in kernel_name:
                    return True

        return False

    def _inject_kernel_loader(
        self, operation: str, kernel_functions: List[Dict]
    ) -> bool:
        """Injeta código para carregar e utilizar o kernel CUDA nativo"""
        # Dependendo da operação, injeta em diferentes pontos do código
        if operation == "matmul":
            # Constrói o código para injeção
            kernel_function = next(
                (f for f in kernel_functions if "matmul" in f["name"].lower()), None
            )
            if not kernel_function:
                return False

            # Substitui a implementação do método _matmul_tensor_cores ou adiciona nova
            self._inject_matmul_implementation(kernel_function)

        elif operation == "layernorm":
            kernel_function = next(
                (
                    f
                    for f in kernel_functions
                    if "layernorm" in f["name"].lower()
                    or "layer_norm" in f["name"].lower()
                ),
                None,
            )
            if not kernel_function:
                return False

            self._inject_layernorm_implementation(kernel_function)

        elif operation == "gelu":
            kernel_function = next(
                (f for f in kernel_functions if "gelu" in f["name"].lower()), None
            )
            if not kernel_function:
                return False

            self._inject_gelu_implementation(kernel_function)

        elif operation == "softmax":
            kernel_function = next(
                (f for f in kernel_functions if "softmax" in f["name"].lower()), None
            )
            if not kernel_function:
                return False

            self._inject_softmax_implementation(kernel_function)

        return True

    def _inject_matmul_implementation(self, kernel_function: Dict) -> bool:
        """Injeta implementação de matmul usando kernel CUDA nativo"""
        # Extrai nome do kernel para utilização
        kernel_name = kernel_function["name"]

        # Verifica se _matmul_tensor_cores já existe no engine
        has_tensor_cores = hasattr(self.engine, "_matmul_tensor_cores")

        # Cria código de implementação otimizada
        implementation = f"""
        def _optimized_matmul_native(self, A: ArrayType, B: ArrayType) -> ArrayType:
            '''Multiplicação de matrizes usando kernel CUDA nativo otimizado ({kernel_name})'''
            try:
                # Garantir tipos corretos e transferir para GPU
                A_gpu = self.to_device(A)
                B_gpu = self.to_device(B)
                
                # Obter dimensões
                if A_gpu.ndim == 1:
                    A_gpu = A_gpu.reshape(1, -1)
                if B_gpu.ndim == 1:
                    B_gpu = B_gpu.reshape(-1, 1)
                
                M, K = A_gpu.shape
                K2, N = B_gpu.shape
                
                # Verificar compatibilidade
                if K != K2:
                    raise ValueError(f"Dimensões incompatíveis: ({{M}}x{{K}}) @ ({{K2}}x{{N}})")
                
                # Alocar resultado
                C_gpu = self.cp.empty((M, N), dtype=A_gpu.dtype)
                
                # Obter ponteiros para arrays
                A_ptr = A_gpu.data.ptr
                B_ptr = B_gpu.data.ptr
                C_ptr = C_gpu.data.ptr
                
                # Configurar grid e blocos
                threads_per_block = (32, 32)
                blocks_per_grid = ((M + threads_per_block[0] - 1) // threads_per_block[0],
                                 (N + threads_per_block[1] - 1) // threads_per_block[1])
                
                # Verificar se o kernel existe ou compilar
                if not hasattr(self, 'matmul_native_kernel'):
                    kernel_code = self._load_kernel_from_file('{self.available_kernels["matmul"]}')
                    self.matmul_native_kernel = self._compile_kernel_safe(kernel_code, '{kernel_name}')
                
                # Lançar kernel
                with self.get_next_stream() as stream:
                    self.matmul_native_kernel(grid=blocks_per_grid, 
                                           block=threads_per_block,
                                           args=(A_ptr, B_ptr, C_ptr, M, N, K),
                                           stream=stream)
                
                # Retornar para CPU se necessário
                if isinstance(A, np.ndarray):
                    return self.cp.asnumpy(C_gpu)
                return C_gpu
                
            except Exception as e:
                self.logger.error(f"Erro em matmul nativo: {{e}}")
                # Fallback para implementação padrão
                if has_tensor_cores:
                    return self._matmul_tensor_cores(A, B)
                else:
                    return self._matmul_cublas(A, B)
        """

        # Adiciona método ao engine
        exec_globals = {"ArrayType": np.ndarray}
        exec(implementation, exec_globals)

        # Patch para o método original (_matmul_tensor_cores ou _matmul_cublas)
        if has_tensor_cores:
            # Salva referência ao método original para fallback
            self.engine._matmul_tensor_cores_original = self.engine._matmul_tensor_cores
            # Substitui com nova implementação
            self.engine._matmul_tensor_cores = exec_globals[
                "_optimized_matmul_native"
            ].__get__(self.engine)
        else:
            # Salva referência ao método _matmul_cublas para fallback
            self.engine._matmul_cublas_original = self.engine._matmul_cublas
            # Substitui com nova implementação
            self.engine._matmul_cublas = exec_globals[
                "_optimized_matmul_native"
            ].__get__(self.engine)

        self.logger.info(
            f"Éter: Kernel CUDA nativo injetado para matmul: {kernel_name}"
        )
        return True

    def _inject_layernorm_implementation(self, kernel_function: Dict) -> bool:
        """Injeta implementação de layer_norm usando kernel CUDA nativo"""
        # Extrai nome do kernel para utilização
        kernel_name = kernel_function["name"]

        # Cria código de implementação otimizada
        implementation = f"""
        def _optimized_layernorm_native(self, x: ArrayType, gamma: ArrayType, beta: ArrayType) -> ArrayType:
            '''Layer normalization usando kernel CUDA nativo otimizado ({kernel_name})'''
            try:
                # Garantir tipos corretos e transferir para GPU
                x_gpu = self.to_device(x)
                gamma_gpu = self.to_device(gamma)
                beta_gpu = self.to_device(beta)
                
                # Obter dimensões
                if x_gpu.ndim < 2:
                    x_gpu = x_gpu.reshape(1, -1)
                
                batch_seq_size = np.prod(x_gpu.shape[:-1]).item()
                hidden_size = x_gpu.shape[-1]
                
                # Alocar resultado
                output_gpu = self.cp.empty_like(x_gpu)
                
                # Obter ponteiros para arrays
                x_ptr = x_gpu.data.ptr
                gamma_ptr = gamma_gpu.data.ptr
                beta_ptr = beta_gpu.data.ptr
                output_ptr = output_gpu.data.ptr
                
                # Configurar grid e blocos
                threads_per_block = min(hidden_size, 1024)
                blocks_per_grid = batch_seq_size
                
                # Verificar se o kernel existe ou compilar
                if not hasattr(self, 'layernorm_native_kernel'):
                    kernel_code = self._load_kernel_from_file('{self.available_kernels["layernorm"]}')
                    self.layernorm_native_kernel = self._compile_kernel_safe(kernel_code, '{kernel_name}')
                
                # Calcular shared memory necessária
                # 2 arrays (média e variância) de floats para cada thread block
                shared_mem_size = 2 * threads_per_block * 4  # 4 bytes por float
                
                # Lançar kernel
                with self.get_next_stream() as stream:
                    self.layernorm_native_kernel(grid=blocks_per_grid, 
                                              block=threads_per_block,
                                              args=(x_ptr, gamma_ptr, beta_ptr, output_ptr, 
                                                    batch_seq_size, hidden_size),
                                              stream=stream,
                                              shared_mem=shared_mem_size)
                
                # Retornar para CPU se necessário
                if isinstance(x, np.ndarray):
                    return self.cp.asnumpy(output_gpu)
                return output_gpu
                
            except Exception as e:
                self.logger.error(f"Erro em layernorm nativo: {{e}}")
                # Fallback para implementação padrão
                return self._optimized_layernorm(x, gamma, beta)
        """

        # Adiciona método ao engine
        exec_globals = {"ArrayType": np.ndarray}
        exec(implementation, exec_globals)

        # Salva referência ao método original para fallback
        self.engine._optimized_layernorm_original = self.engine._optimized_layernorm
        # Substitui com nova implementação
        self.engine._optimized_layernorm = exec_globals[
            "_optimized_layernorm_native"
        ].__get__(self.engine)

        self.logger.info(
            f"Éter: Kernel CUDA nativo injetado para layernorm: {kernel_name}"
        )
        return True

    def _inject_gelu_implementation(self, kernel_function: Dict) -> bool:
        """Injeta implementação de gelu usando kernel CUDA nativo"""
        # Extrai nome do kernel para utilização
        kernel_name = kernel_function["name"]

        # Cria código de implementação otimizada
        implementation = f"""
        def _optimized_gelu_native(self, x: ArrayType) -> ArrayType:
            '''GELU usando kernel CUDA nativo otimizado ({kernel_name})'''
            try:
                # Garantir tipos corretos e transferir para GPU
                x_gpu = self.to_device(x)
                
                # Alocar resultado
                output_gpu = self.cp.empty_like(x_gpu)
                
                # Obter ponteiros para arrays
                x_ptr = x_gpu.data.ptr
                output_ptr = output_gpu.data.ptr
                
                # Obter tamanho total
                total_size = x_gpu.size
                
                # Configurar grid e blocos
                threads_per_block = min(total_size, 1024)
                blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block
                
                # Verificar se o kernel existe ou compilar
                if not hasattr(self, 'gelu_native_kernel'):
                    kernel_code = self._load_kernel_from_file('{self.available_kernels["gelu"]}')
                    self.gelu_native_kernel = self._compile_kernel_safe(kernel_code, '{kernel_name}')
                
                # Lançar kernel
                with self.get_next_stream() as stream:
                    self.gelu_native_kernel(grid=blocks_per_grid, 
                                         block=threads_per_block,
                                         args=(x_ptr, output_ptr, total_size),
                                         stream=stream)
                
                # Retornar para CPU se necessário
                if isinstance(x, np.ndarray):
                    return self.cp.asnumpy(output_gpu)
                return output_gpu
                
            except Exception as e:
                self.logger.error(f"Erro em gelu nativo: {{e}}")
                # Fallback para implementação padrão
                return self._optimized_gelu(x)
        """

        # Adiciona método ao engine
        exec_globals = {"ArrayType": np.ndarray}
        exec(implementation, exec_globals)

        # Salva referência ao método original para fallback
        self.engine._optimized_gelu_original = self.engine._optimized_gelu
        # Substitui com nova implementação
        self.engine._optimized_gelu = exec_globals["_optimized_gelu_native"].__get__(
            self.engine
        )

        self.logger.info(f"Éter: Kernel CUDA nativo injetado para gelu: {kernel_name}")
        return True

    def _inject_softmax_implementation(self, kernel_function: Dict) -> bool:
        """Injeta implementação de softmax usando kernel CUDA nativo"""
        # Extrai nome do kernel para utilização
        kernel_name = kernel_function["name"]

        # Cria código de implementação otimizada
        implementation = f"""
        def _optimized_softmax_native(self, x: ArrayType, axis: int = -1) -> ArrayType:
            '''Softmax usando kernel CUDA nativo otimizado ({kernel_name})'''
            try:
                # Garantir tipos corretos e transferir para GPU
                x_gpu = self.to_device(x)
                
                # Tratamento de eixo negativo
                if axis < 0:
                    axis = x_gpu.ndim + axis
                
                # Precisa reorganizar dados se o eixo não for o último (mais eficiente)
                if axis != x_gpu.ndim - 1:
                    # Reorganiza dimensões para colocar o eixo desejado por último
                    perm = list(range(x_gpu.ndim))
                    perm.pop(axis)
                    perm.append(axis)
                    x_t = self.cp.transpose(x_gpu, perm)
                    
                    # Processa com softmax
                    result_t = self._softmax_last_dim_native(x_t)
                    
                    # Reorganiza de volta
                    inv_perm = [0] * x_gpu.ndim
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                    result = self.cp.transpose(result_t, inv_perm)
                else:
                    # Eixo já é o último, mais eficiente
                    result = self._softmax_last_dim_native(x_gpu)
                
                # Retornar para CPU se necessário
                if isinstance(x, np.ndarray):
                    return self.cp.asnumpy(result)
                return result
                
            except Exception as e:
                self.logger.error(f"Erro em softmax nativo: {{e}}")
                # Fallback para implementação padrão
                return self._optimized_softmax(x, axis)
        
        def _softmax_last_dim_native(self, x: ArrayType) -> ArrayType:
            '''Implementação de softmax para o último eixo usando CUDA nativo'''
            # Obter dimensões
            shape = x.shape
            rows = np.prod(shape[:-1]).item() if len(shape) > 1 else 1
            cols = shape[-1]
            
            # Alocar resultado
            output = self.cp.empty_like(x)
            
            # Obter ponteiros para arrays
            x_ptr = x.data.ptr
            output_ptr = output.data.ptr
            
            # Configurar grid e blocos
            threads_per_block = min(cols, 1024)
            blocks_per_grid = rows
            
            # Verificar se o kernel existe ou compilar
            if not hasattr(self, 'softmax_native_kernel'):
                kernel_code = self._load_kernel_from_file('{self.available_kernels["softmax"]}')
                self.softmax_native_kernel = self._compile_kernel_safe(kernel_code, '{kernel_name}')
            
            # Calcular shared memory necessária
            shared_mem_size = threads_per_block * 4  # 4 bytes por float
            
            # Lançar kernel
            with self.get_next_stream() as stream:
                self.softmax_native_kernel(grid=blocks_per_grid, 
                                        block=threads_per_block,
                                        args=(x_ptr, output_ptr, rows, cols),
                                        stream=stream,
                                        shared_mem=shared_mem_size)
            
            return output
        """

        # Adiciona método ao engine
        exec_globals = {"ArrayType": np.ndarray}
        exec(implementation, exec_globals)

        # Adiciona método auxiliar ao engine
        self.engine._softmax_last_dim_native = exec_globals[
            "_softmax_last_dim_native"
        ].__get__(self.engine)

        # Salva referência ao método original para fallback
        self.engine._optimized_softmax_original = self.engine._optimized_softmax
        # Substitui com nova implementação
        self.engine._optimized_softmax = exec_globals[
            "_optimized_softmax_native"
        ].__get__(self.engine)

        self.logger.info(
            f"Éter: Kernel CUDA nativo injetado para softmax: {kernel_name}"
        )
        return True

    def _benchmark_current_implementation(
        self, operation: str, dtype=np.float32
    ) -> float:
        """Realiza benchmark da implementação atual"""
        if operation in self.benchmark_results:
            return self.benchmark_results[operation]

        # Tamanhos padrão para benchmark
        if operation == "matmul":
            A = np.random.random((1024, 1024)).astype(dtype)
            B = np.random.random((1024, 1024)).astype(dtype)

            # Warmup
            for _ in range(3):
                self.engine.matmul(A, B)

            # Benchmark
            start = time.time()
            for _ in range(10):
                self.engine.matmul(A, B)
            avg_time = (time.time() - start) / 10

        elif operation == "layernorm":
            x = np.random.random((1024, 1024)).astype(dtype)
            gamma = np.ones((1024,), dtype=dtype)
            beta = np.zeros((1024,), dtype=dtype)

            # Warmup
            for _ in range(3):
                self.engine.layer_norm(x, gamma, beta)

            # Benchmark
            start = time.time()
            for _ in range(10):
                self.engine.layer_norm(x, gamma, beta)
            avg_time = (time.time() - start) / 10

        elif operation == "gelu":
            x = np.random.random((1024, 1024)).astype(dtype)

            # Warmup
            for _ in range(3):
                self.engine.gelu(x)

            # Benchmark
            start = time.time()
            for _ in range(10):
                self.engine.gelu(x)
            avg_time = (time.time() - start) / 10

        elif operation == "softmax":
            x = np.random.random((1024, 1024)).astype(dtype)

            # Warmup
            for _ in range(3):
                self.engine.softmax(x)

            # Benchmark
            start = time.time()
            for _ in range(10):
                self.engine.softmax(x)
            avg_time = (time.time() - start) / 10

        else:
            return 0.0

        self.benchmark_results[operation] = avg_time
        return avg_time

    def _load_kernel_from_file(self, file_path: str) -> str:
        """Carrega código fonte do kernel a partir do arquivo"""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Erro ao carregar kernel: {e}")
            return ""

    def accelerate_fused_operations(self):
        """Ativa operações fundidas para máxima performance"""
        # Se tivermos kernel para operações fundidas
        if "fused_ops" in self.available_kernels:
            try:
                kernel_path = self.available_kernels["fused_ops"]

                # Carrega kernel de fusão
                with open(kernel_path, "r") as f:
                    kernel_code = f.read()

                # Procura por kernels fundidos específicos
                fused_kernels = []

                if "fused_gelu_layernorm" in kernel_code.lower():
                    fused_kernels.append("gelu_layernorm")

                if "fused_matmul_gelu" in kernel_code.lower():
                    fused_kernels.append("matmul_gelu")

                if "fused_attention_softmax" in kernel_code.lower():
                    fused_kernels.append("attention_softmax")

                # Ativa cada kernel fundido
                for fused_op in fused_kernels:
                    self._activate_fused_kernel(fused_op, kernel_code)

                self.logger.info(
                    f"Éter: Ativados {len(fused_kernels)} kernels fundidos para máxima performance"
                )

            except Exception as e:
                self.logger.error(f"Erro ao ativar operações fundidas: {str(e)}")

    def _activate_fused_kernel(self, fused_op: str, kernel_code: str):
        """Ativa um kernel fundido específico"""
        # Implementação específica para cada tipo de fusão
        if fused_op == "gelu_layernorm":
            # Injeta método para operação fundida GELU+LayerNorm
            self._inject_fused_gelu_layernorm(kernel_code)

        elif fused_op == "matmul_gelu":
            # Injeta método para operação fundida MatMul+GELU
            self._inject_fused_matmul_gelu(kernel_code)

        elif fused_op == "attention_softmax":
            # Injeta método para operação fundida Attention+Softmax
            self._inject_fused_attention_softmax(kernel_code)

    def _inject_fused_gelu_layernorm(self, kernel_code: str):
        """Injeta implementação de operação fundida GELU+LayerNorm"""
        # Busca pelo nome do kernel no código
        kernel_name = None
        for line in kernel_code.split("\n"):
            if "void" in line and "fused_gelu_layernorm" in line:
                kernel_name = line.split("void ")[1].split("(")[0].strip()
                break

        if not kernel_name:
            self.logger.warning("Kernel fused_gelu_layernorm não encontrado no código")
            return False

        # Cria função de fusão no engine
        self.engine.fused_gelu_layernorm = self._create_fused_gelu_layernorm_method(
            kernel_name
        )
        self.logger.info(f"Éter: Kernel fundido ativado - {kernel_name}")

    def _create_fused_gelu_layernorm_method(self, kernel_name):
        """Cria método para GELU+LayerNorm fundidos"""

        def fused_gelu_layernorm(self, x, gamma, beta):
            """Executa GELU seguido de LayerNorm em uma única operação fundida"""
            try:
                # Garantir tipos corretos e transferir para GPU
                x_gpu = self.to_device(x)
                gamma_gpu = self.to_device(gamma)
                beta_gpu = self.to_device(beta)

                # Obter dimensões
                if x_gpu.ndim < 2:
                    x_gpu = x_gpu.reshape(1, -1)

                batch_size = 1
                seq_len = 1
                hidden_size = x_gpu.shape[-1]

                if x_gpu.ndim >= 3:
                    batch_size = x_gpu.shape[0]
                    seq_len = x_gpu.shape[1]
                elif x_gpu.ndim == 2:
                    batch_size = x_gpu.shape[0]

                # Alocar resultado
                output_gpu = self.cp.empty_like(x_gpu)

                # Obter ponteiros para arrays
                x_ptr = x_gpu.data.ptr
                gamma_ptr = gamma_gpu.data.ptr
                beta_ptr = beta_gpu.data.ptr
                output_ptr = output_gpu.data.ptr

                # Configurar grid e blocos
                threads_per_block = min(hidden_size, 1024)
                blocks_per_grid = batch_size * seq_len

                # Verificar se o kernel existe ou compilar
                if not hasattr(self, "fused_gelu_layernorm_kernel"):
                    kernel_code = self._load_kernel_from_file(
                        self.available_kernels["fused_ops"]
                    )
                    self.fused_gelu_layernorm_kernel = self._compile_kernel_safe(
                        kernel_code, kernel_name
                    )

                # Calcular shared memory necessária
                shared_mem_size = 2 * threads_per_block * 4  # 4 bytes por float

                # Lançar kernel
                with self.get_next_stream() as stream:
                    self.fused_gelu_layernorm_kernel(
                        grid=blocks_per_grid,
                        block=threads_per_block,
                        args=(
                            x_ptr,
                            gamma_ptr,
                            beta_ptr,
                            output_ptr,
                            batch_size,
                            seq_len,
                            hidden_size,
                        ),
                        stream=stream,
                        shared_mem=shared_mem_size,
                    )

                # Retornar para CPU se necessário
                if isinstance(x, np.ndarray):
                    return self.cp.asnumpy(output_gpu)
                return output_gpu

            except Exception as e:
                self.logger.error(f"Erro em fused_gelu_layernorm: {e}")
                # Fallback para operações separadas
                gelu_out = self.gelu(x)
                return self.layer_norm(gelu_out, gamma, beta)

        return fused_gelu_layernorm.__get__(self.engine)

    def _inject_fused_matmul_gelu(self, kernel_code: str):
        # Implementação similar à _inject_fused_gelu_layernorm
        pass

    def _inject_fused_attention_softmax(self, kernel_code: str):
        # Implementação similar à _inject_fused_gelu_layernorm
        pass

    def get_acceleration_report(self) -> Dict:
        """Retorna relatório detalhado sobre acelerações ativas"""
        report = {
            "active_accelerations": sum(
                1 for op in self.acceleration_stats.values() if op["status"] == "active"
            ),
            "total_kernels": len(self.available_kernels),
            "operations": self.acceleration_stats,
            "fused_operations": [
                k for k in self.loaded_kernels.keys() if "fused" in k.lower()
            ],
            "average_speedup": 0.0,
        }

        # Calcula speedup médio
        speedups = [
            op["speedup"]
            for op in self.acceleration_stats.values()
            if op["status"] == "active"
        ]
        if speedups:
            report["average_speedup"] = sum(speedups) / len(speedups)

        return report

    def matmul(self, a, b, out=None, use_tensor_cores=True, measure_performance=False):
        """
        Executa multiplicação de matrizes usando kernels CUDA otimizados

        Args:
            a: tensor de entrada 1
            b: tensor de entrada 2
            out: tensor de saída opcional
            use_tensor_cores: se True, usa Tensor Cores se disponíveis
            measure_performance: se True, mede o desempenho da operação

        Returns:
            tensor resultante da multiplicação
        """
        operation = "matmul"
        start_time = time.time() if measure_performance else None
        success = False

        if not self.is_initialized:
            self.logger.debug("CUDA não inicializado, usando fallback para matmul")
            result = self._fallback_matmul(a, b, out)
            return result

        try:
            # Verifica se as dimensões são compatíveis
            if a.ndim < 2 or b.ndim < 2:
                raise ValueError(
                    "Entradas para matmul devem ter pelo menos 2 dimensões"
                )

            if a.shape[-1] != b.shape[-2]:
                raise ValueError(
                    f"Dimensões incompatíveis para matmul: {a.shape} e {b.shape}"
                )

            # Seleciona kernel apropriado baseado nos tipos de dados e disponibilidade
            use_tc = use_tensor_cores and self.has_tensor_cores

            # Converte para arrays contíguos se necessário
            a_cont = np.ascontiguousarray(a)
            b_cont = np.ascontiguousarray(b)

            # Prepara o tensor de saída se não fornecido
            if out is None:
                out_shape = list(a.shape[:-1]) + [b.shape[-1]]
                out = np.empty(out_shape, dtype=np.float32)

            # Chama o kernel apropriado
            if use_tc and hasattr(self.lib, "matmul_tensor_cores"):
                self.lib.matmul_tensor_cores(
                    a_cont.ctypes.data_as(ctypes.c_void_p),
                    b_cont.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(a.shape[0]),
                    ctypes.c_int(a.shape[1]),
                    ctypes.c_int(b.shape[1]),
                )
            else:
                self.lib.matmul(
                    a_cont.ctypes.data_as(ctypes.c_void_p),
                    b_cont.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(a.shape[0]),
                    ctypes.c_int(a.shape[1]),
                    ctypes.c_int(b.shape[1]),
                )

            success = True
            if measure_performance:
                end_time = time.time()
                self._update_stats(operation, success, end_time - start_time)

            return out

        except Exception as e:
            self.logger.warning(f"Erro ao executar matmul CUDA: {e}")
            if measure_performance:
                end_time = time.time()
                self._update_stats(operation, False, end_time - start_time)

            # Fallback para implementação do engine ou numpy
            return self._fallback_matmul(a, b, out)

    def _fallback_matmul(self, a, b, out=None):
        """Método fallback para matmul quando CUDA falha"""
        if self.engine and hasattr(self.engine, "matmul"):
            self.logger.debug("Usando fallback do engine para matmul")
            return self.engine.matmul(a, b, out=out)
        else:
            self.logger.debug("Usando fallback do NumPy para matmul")
            return np.matmul(a, b, out=out)

    def layer_norm(
        self, x, gamma=None, beta=None, epsilon=1e-5, measure_performance=False
    ):
        """
        Executa layer normalization usando kernels CUDA otimizados

        Args:
            x: tensor de entrada
            gamma: parâmetro de escala (opcional)
            beta: parâmetro de deslocamento (opcional)
            epsilon: valor para estabilidade numérica
            measure_performance: se True, mede o desempenho da operação

        Returns:
            tensor normalizado
        """
        operation = "layer_norm"
        start_time = time.time() if measure_performance else None
        success = False

        if not self.is_initialized:
            self.logger.debug("CUDA não inicializado, usando fallback para layer_norm")
            return self._fallback_layer_norm(x, gamma, beta, epsilon)

        try:
            # Prepara parâmetros
            x_cont = np.ascontiguousarray(x)
            last_dim = x.shape[-1]

            # Cria gamma e beta se não fornecidos
            if gamma is None:
                gamma = np.ones(last_dim, dtype=x.dtype)
            if beta is None:
                beta = np.zeros(last_dim, dtype=x.dtype)

            gamma_cont = np.ascontiguousarray(gamma)
            beta_cont = np.ascontiguousarray(beta)

            # Aloca tensor de saída
            out = np.empty_like(x)

            # Chama o kernel apropriado
            if hasattr(self.lib, "layer_norm"):
                self.lib.layer_norm(
                    x_cont.ctypes.data_as(ctypes.c_void_p),
                    gamma_cont.ctypes.data_as(ctypes.c_void_p),
                    beta_cont.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(x.size // last_dim),  # batch_size
                    ctypes.c_int(last_dim),  # hidden_size
                    ctypes.c_float(epsilon),
                )

                success = True
                if measure_performance:
                    end_time = time.time()
                    self._update_stats(operation, success, end_time - start_time)

                return out
            else:
                raise AttributeError(
                    "Função layer_norm não disponível na biblioteca CUDA"
                )

        except Exception as e:
            self.logger.warning(f"Erro ao executar layer_norm CUDA: {e}")
            if measure_performance:
                end_time = time.time()
                self._update_stats(operation, False, end_time - start_time)

            # Fallback para implementação do engine ou manual
            return self._fallback_layer_norm(x, gamma, beta, epsilon)

    def _fallback_layer_norm(self, x, gamma=None, beta=None, epsilon=1e-5):
        """Método fallback para layer_norm quando CUDA falha"""
        if self.engine and hasattr(self.engine, "layer_norm"):
            self.logger.debug("Usando fallback do engine para layer_norm")
            return self.engine.layer_norm(x, gamma, beta, epsilon)
        else:
            self.logger.debug("Usando implementação manual para layer_norm")
            # Implementação manual de layer_norm
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            normed = (x - mean) / np.sqrt(var + epsilon)

            if gamma is not None and beta is not None:
                return gamma * normed + beta
            elif gamma is not None:
                return gamma * normed
            elif beta is not None:
                return normed + beta
            else:
                return normed

    def gelu(self, x, measure_performance=False):
        """
        Aplica função de ativação GELU usando kernels CUDA otimizados

        Args:
            x: tensor de entrada
            measure_performance: se True, mede o desempenho da operação

        Returns:
            tensor após aplicação do GELU
        """
        operation = "gelu"
        start_time = time.time() if measure_performance else None
        success = False

        if not self.is_initialized:
            self.logger.debug("CUDA não inicializado, usando fallback para gelu")
            return self._fallback_gelu(x)

        try:
            # Prepara tensor
            x_cont = np.ascontiguousarray(x)

            # Aloca tensor de saída
            out = np.empty_like(x)

            # Chama o kernel
            if hasattr(self.lib, "gelu"):
                self.lib.gelu(
                    x_cont.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(x.size),
                )

                success = True
                if measure_performance:
                    end_time = time.time()
                    self._update_stats(operation, success, end_time - start_time)

                return out
            else:
                raise AttributeError("Função gelu não disponível na biblioteca CUDA")

        except Exception as e:
            self.logger.warning(f"Erro ao executar gelu CUDA: {e}")
            if measure_performance:
                end_time = time.time()
                self._update_stats(operation, False, end_time - start_time)

            # Fallback para implementação do engine ou manual
            return self._fallback_gelu(x)

    def _fallback_gelu(self, x):
        """Método fallback para gelu quando CUDA falha"""
        if self.engine and hasattr(self.engine, "gelu"):
            self.logger.debug("Usando fallback do engine para gelu")
            return self.engine.gelu(x)
        else:
            self.logger.debug("Usando implementação manual para gelu")
            # Implementação aproximada de GELU
            return (
                0.5
                * x
                * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
            )

    def softmax(self, tensor, axis=-1, measure_performance=False):
        """
        Aplica a função softmax ao tensor de entrada.

        Args:
            tensor: Tensor de entrada
            axis: Eixo ao longo do qual aplicar softmax
            measure_performance: Se True, mede o tempo de execução

        Returns:
            Tensor com softmax aplicado
        """
        start_time = time.time() if measure_performance else None

        try:
            # Se o tensor não estiver na GPU, mova-o
            if not self._is_gpu_tensor(tensor):
                tensor = cp.asarray(tensor)

            # Aplicar softmax usando CUDA
            result = cp.zeros_like(tensor)

            # Configurar dimensões para o kernel
            if axis == -1 or axis == tensor.ndim - 1:
                # Caso comum: softmax no último eixo
                rows = tensor.size // tensor.shape[-1]
                cols = tensor.shape[-1]

                # Reshape para 2D para o kernel CUDA
                input_reshaped = tensor.reshape(rows, cols)
                output_reshaped = result.reshape(rows, cols)

                # Chamar kernel softmax
                self._call_cuda_kernel(
                    "softmax",
                    input_reshaped.data.ptr,
                    output_reshaped.data.ptr,
                    rows,
                    cols,
                )

                # Incrementar contador de operações com sucesso
                self._successful_ops += 1
            else:
                # Para outros eixos, usar a implementação padrão
                result = cp.softmax(tensor, axis=axis)

            if measure_performance:
                self._op_times["softmax"] = (time.time() - start_time) * 1000  # ms

            return result

        except Exception as e:
            self._log_error(f"Erro no softmax: {str(e)}")
            self._failed_ops += 1

            # Fallback para softmax padrão do CuPy
            result = cp.softmax(tensor, axis=axis)

            if measure_performance:
                self._op_times["softmax"] = (time.time() - start_time) * 1000  # ms

            return result

    def layer_norm_gelu(
        self, input_tensor, gamma, beta, epsilon=1e-5, measure_performance=False
    ):
        """
        Aplica layer normalization seguido por GELU usando kernel fundido

        Args:
            input_tensor: Tensor de entrada (batch_size, hidden_size)
            gamma: Parâmetro de escala (hidden_size,)
            beta: Parâmetro de deslocamento (hidden_size,)
            epsilon: Parâmetro de estabilidade numérica
            measure_performance: Se True, mede o tempo de execução

        Returns:
            Tensor com layer norm e GELU aplicados
        """
        start_time = time.time() if measure_performance else None

        try:
            # Verificar se tensores estão na GPU
            if not self._is_gpu_tensor(input_tensor):
                input_tensor = cp.asarray(input_tensor)
            if not self._is_gpu_tensor(gamma):
                gamma = cp.asarray(gamma)
            if not self._is_gpu_tensor(beta):
                beta = cp.asarray(beta)

            # Verificar dimensões
            batch_size, hidden_size = input_tensor.shape
            if gamma.shape[0] != hidden_size or beta.shape[0] != hidden_size:
                raise ValueError(
                    f"Dimensões incompatíveis: gamma e beta devem ter formato ({hidden_size},)"
                )

            # Preparar tensor de saída
            output = cp.zeros_like(input_tensor)

            # Usar kernel FP16 se disponível e tensor for half
            if self.use_fp16 and input_tensor.dtype == cp.float16:
                self._call_cuda_kernel(
                    "layer_norm_gelu_half",
                    input_tensor.data.ptr,
                    gamma.data.ptr,
                    beta.data.ptr,
                    output.data.ptr,
                    batch_size,
                    hidden_size,
                    epsilon,
                )
            else:
                # Usar kernel padrão float32
                self._call_cuda_kernel(
                    "layer_norm_gelu",
                    input_tensor.data.ptr,
                    gamma.data.ptr,
                    beta.data.ptr,
                    output.data.ptr,
                    batch_size,
                    hidden_size,
                    epsilon,
                )

            # Incrementar contador de operações com sucesso
            self._successful_ops += 1

            if measure_performance:
                self._op_times["layer_norm_gelu"] = (
                    time.time() - start_time
                ) * 1000  # ms

            return output

        except Exception as e:
            self._log_error(f"Erro na layer_norm_gelu fundida: {str(e)}")
            self._failed_ops += 1

            # Fallback para execução separada
            result = self._fallback_layer_norm_gelu(input_tensor, gamma, beta, epsilon)

            if measure_performance:
                self._op_times["layer_norm_gelu"] = (
                    time.time() - start_time
                ) * 1000  # ms

            return result

    def _fallback_layer_norm_gelu(self, input_tensor, gamma, beta, epsilon=1e-5):
        """
        Fallback para layer norm seguido por GELU quando o kernel fundido falha
        """
        # Primeiro executar layer norm
        mean = cp.mean(input_tensor, axis=-1, keepdims=True)
        var = cp.var(input_tensor, axis=-1, keepdims=True)
        normalized = (input_tensor - mean) / cp.sqrt(var + epsilon)
        scaled = gamma * normalized + beta

        # Depois aplicar GELU
        return self.gelu(scaled)

    def softmax_layer_norm(
        self, input_tensor, gamma, beta, epsilon=1e-5, measure_performance=False
    ):
        """
        Executa softmax seguido por layer normalization usando kernel fundido

        Args:
            input_tensor: Tensor de entrada (batch_size, seq_len)
            gamma: Parâmetro de escala (seq_len,)
            beta: Parâmetro de deslocamento (seq_len,)
            epsilon: Parâmetro de estabilidade numérica
            measure_performance: Se True, mede o tempo de execução

        Returns:
            Tensor com softmax e layer norm aplicados
        """
        start_time = time.time() if measure_performance else None

        try:
            # Verificar se tensores estão na GPU
            if not self._is_gpu_tensor(input_tensor):
                input_tensor = cp.asarray(input_tensor)
            if not self._is_gpu_tensor(gamma):
                gamma = cp.asarray(gamma)
            if not self._is_gpu_tensor(beta):
                beta = cp.asarray(beta)

            # Verificar dimensões
            if input_tensor.ndim != 2:
                raise ValueError(
                    f"Entrada deve ser 2D, mas tem formato {input_tensor.shape}"
                )

            rows, cols = input_tensor.shape
            if gamma.shape[0] != cols or beta.shape[0] != cols:
                raise ValueError(
                    f"Dimensões incompatíveis: gamma e beta devem ter formato ({cols},)"
                )

            # Preparar tensor de saída
            output = cp.zeros_like(input_tensor)

            # Usar kernel FP16 se disponível e tensor for half
            if self.use_fp16 and input_tensor.dtype == cp.float16:
                self._call_cuda_kernel(
                    "softmax_layer_norm_half",
                    input_tensor.data.ptr,
                    gamma.data.ptr,
                    beta.data.ptr,
                    output.data.ptr,
                    rows,
                    cols,
                    epsilon,
                )
            else:
                # Usar kernel padrão float32
                self._call_cuda_kernel(
                    "softmax_layer_norm",
                    input_tensor.data.ptr,
                    gamma.data.ptr,
                    beta.data.ptr,
                    output.data.ptr,
                    rows,
                    cols,
                    epsilon,
                )

            # Incrementar contador de operações com sucesso
            self._successful_ops += 1

            if measure_performance:
                self._op_times["softmax_layer_norm"] = (
                    time.time() - start_time
                ) * 1000  # ms

            return output

        except Exception as e:
            self._log_error(f"Erro na softmax_layer_norm fundida: {str(e)}")
            self._failed_ops += 1

            # Fallback para execução separada
            result = self._fallback_softmax_layer_norm(
                input_tensor, gamma, beta, epsilon
            )

            if measure_performance:
                self._op_times["softmax_layer_norm"] = (
                    time.time() - start_time
                ) * 1000  # ms

            return result

    def _fallback_softmax_layer_norm(self, input_tensor, gamma, beta, epsilon=1e-5):
        """
        Fallback para softmax seguido por layer norm quando o kernel fundido falha
        """
        # Primeiro aplicar softmax
        softmax_result = self.softmax(input_tensor, axis=-1)

        # Depois executar layer norm
        mean = cp.mean(softmax_result, axis=-1, keepdims=True)
        var = cp.var(softmax_result, axis=-1, keepdims=True)
        normalized = (softmax_result - mean) / cp.sqrt(var + epsilon)

        return gamma * normalized + beta

    def auto_optimize(
        self, input_shape=None, batch_size=None, hidden_size=None, seq_len=None
    ):
        """
        Configura automaticamente o acelerador CUDA para o melhor desempenho

        Args:
            input_shape: forma dos tensores de entrada (opcional)
            batch_size: tamanho do lote (opcional)
            hidden_size: dimensão oculta dos tensores (opcional)
            seq_len: comprimento da sequência (opcional)

        Returns:
            dicionário com as configurações otimizadas
        """
        self.logger.info("Iniciando otimização automática do acelerador CUDA")

        # Detecta se temos GPU e suas capacidades
        has_gpu = self.is_initialized and self.lib is not None

        # Se não tivermos GPU, não podemos otimizar
        if not has_gpu:
            self.logger.warning("GPU não disponível, operando em modo CPU")
            return {
                "status": "gpu_unavailable",
                "recommended_config": {
                    "use_cuda": False,
                    "use_tensor_cores": False,
                    "use_fp16": False,
                },
            }

        # Detecta capacidades da GPU
        self._detect_gpu_capabilities()

        # Prepara configurações para teste
        configs = {
            "use_tensor_cores": self.has_tensor_cores,
            "use_fp16": self.has_fp16_support,
            "fused_operations": hasattr(self.lib, "layer_norm_gelu")
            or hasattr(self.lib, "softmax_layer_norm"),
        }

        # Executa benchmarks para testar desempenho
        benchmark_results = self._run_benchmarks(
            batch_size=batch_size or 8,
            hidden_size=hidden_size or 768,
            seq_len=seq_len or 128,
        )

        # Seleciona a melhor configuração com base nos benchmarks
        recommended_config = self._select_best_config(benchmark_results)

        self.logger.info(
            f"Otimização automática concluída. Configuração recomendada: {recommended_config}"
        )

        return {
            "status": "success",
            "gpu_capabilities": {
                "has_tensor_cores": self.has_tensor_cores,
                "has_fp16_support": self.has_fp16_support,
            },
            "benchmark_results": benchmark_results,
            "recommended_config": recommended_config,
        }

    def _run_benchmarks(self, batch_size, hidden_size, seq_len):
        """
        Executa benchmarks para medir desempenho com diferentes configurações

        Args:
            batch_size: tamanho do lote para testes
            hidden_size: dimensão oculta para testes
            seq_len: comprimento da sequência para testes

        Returns:
            resultados dos benchmarks
        """
        results = {}

        # Cria tensores aleatórios para teste
        try:
            # Matmul benchmark
            a = np.random.random((batch_size, hidden_size)).astype(np.float32)
            b = np.random.random((hidden_size, hidden_size)).astype(np.float32)

            # Com Tensor Cores
            if self.has_tensor_cores:
                start_time = time.time()
                for _ in range(10):
                    self.matmul(a, b, use_tensor_cores=True)
                tc_time = (time.time() - start_time) / 10 * 1000  # ms
                results["matmul_tensor_cores"] = tc_time

            # Sem Tensor Cores
            start_time = time.time()
            for _ in range(10):
                self.matmul(a, b, use_tensor_cores=False)
            normal_time = (time.time() - start_time) / 10 * 1000  # ms
            results["matmul_normal"] = normal_time

            # Layer Norm benchmark
            x = np.random.random((batch_size, seq_len, hidden_size)).astype(np.float32)
            gamma = np.random.random(hidden_size).astype(np.float32)
            beta = np.random.random(hidden_size).astype(np.float32)

            start_time = time.time()
            for _ in range(10):
                self.layer_norm(x, gamma, beta)
            ln_time = (time.time() - start_time) / 10 * 1000  # ms
            results["layer_norm"] = ln_time

            # GELU benchmark
            start_time = time.time()
            for _ in range(10):
                self.gelu(x)
            gelu_time = (time.time() - start_time) / 10 * 1000  # ms
            results["gelu"] = gelu_time

            # Softmax benchmark
            start_time = time.time()
            for _ in range(10):
                self.softmax(x)
            softmax_time = (time.time() - start_time) / 10 * 1000  # ms
            results["softmax"] = softmax_time

            # Layer Norm + GELU (separado vs fusionado)
            start_time = time.time()
            for _ in range(10):
                norm = self.layer_norm(x, gamma, beta)
                self.gelu(norm)
            separate_time = (time.time() - start_time) / 10 * 1000  # ms
            results["layer_norm_gelu_separate"] = separate_time

            if hasattr(self.lib, "layer_norm_gelu"):
                start_time = time.time()
                for _ in range(10):
                    self.layer_norm_gelu(x, gamma, beta)
                fused_time = (time.time() - start_time) / 10 * 1000  # ms
                results["layer_norm_gelu_fused"] = fused_time

                # Calcula ganho de desempenho
                if fused_time > 0 and separate_time > 0:
                    speedup = separate_time / fused_time
                    results["layer_norm_gelu_speedup"] = speedup

        except Exception as e:
            self.logger.warning(f"Erro durante benchmark: {e}")

        return results

    def _select_best_config(self, benchmark_results):
        """
        Seleciona a melhor configuração com base nos resultados do benchmark

        Args:
            benchmark_results: resultados do benchmark

        Returns:
            dicionário com a melhor configuração
        """
        config = {
            "use_cuda": True,
            "use_tensor_cores": False,
            "use_fp16": False,
            "use_fused_operations": False,
        }

        # Decide sobre uso de Tensor Cores
        if (
            self.has_tensor_cores
            and "matmul_tensor_cores" in benchmark_results
            and "matmul_normal" in benchmark_results
        ):
            tc_time = benchmark_results["matmul_tensor_cores"]
            normal_time = benchmark_results["matmul_normal"]

            if tc_time < normal_time:
                config["use_tensor_cores"] = True
                self.logger.info(
                    f"Tensor Cores proporcionam melhor desempenho: {normal_time/tc_time:.2f}x mais rápido"
                )

        # Decide sobre uso de operações fundidas
        if (
            "layer_norm_gelu_fused" in benchmark_results
            and "layer_norm_gelu_separate" in benchmark_results
        ):
            fused_time = benchmark_results["layer_norm_gelu_fused"]
            separate_time = benchmark_results["layer_norm_gelu_separate"]

            if fused_time < separate_time:
                config["use_fused_operations"] = True
                self.logger.info(
                    f"Operações fundidas proporcionam melhor desempenho: {separate_time/fused_time:.2f}x mais rápido"
                )

        # Decide sobre uso de FP16
        config["use_fp16"] = self.has_fp16_support

        return config


# Função de integração do acelerador ao Éter
def integrate_cuda_accelerator(engine):
    """Integra o acelerador CUDA nativo ao Éter"""
    # Inicializa acelerador
    accelerator = EterCUDAAccelerator(engine)

    # Ativa aceleração para operações principais
    accelerator.accelerate_operations()

    # Ativa operações fundidas para máxima performance
    accelerator.accelerate_fused_operations()

    # Gera relatório de aceleração
    report = accelerator.get_acceleration_report()
    engine.logger.info(
        f"Éter: Aceleração CUDA nativa ativada com speedup médio de {report['average_speedup']:.2f}x"
    )

    # Retorna acelerador para referência futura
    return accelerator
