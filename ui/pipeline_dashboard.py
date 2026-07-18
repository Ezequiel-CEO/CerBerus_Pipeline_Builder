#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard para visualização e gerenciamento de pipelines.

Interface para monitorar e controlar os pipelines de treinamento.
"""

import os
import json
import time
import datetime
from typing import Dict, List, Optional, Any, Tuple
import webbrowser
from pathlib import Path

from cerberus_api.pipeline_builder.core import PipelineBuilder
from cerberus_api.pipeline_builder.models import (
    Pipeline,
    PipelineStatus,
    DataSource,
    ContentCategory,
)
from cerberus_api.utils.logging_config import APILogger

# Importar bibliotecas para interface gráfica
try:
    import gradio as gr
except ImportError:
    gr = None

# Configurar logger
logger = APILogger("pipeline_dashboard")


class PipelineDashboard:
    """
    Dashboard para visualização e gerenciamento de pipelines.
    """

    def __init__(self, base_dir: str = "data", port: int = 7860):
        """
        Inicializa o dashboard.

        Args:
            base_dir: Diretório base para dados.
            port: Porta para o servidor web.
        """
        self.base_dir = base_dir
        self.port = port
        self.pipeline_builder = PipelineBuilder(base_dir=base_dir)

        # Verificar se Gradio está disponível
        if gr is None:
            logger.warning("Gradio não está instalado. O dashboard será limitado.")
            self.has_gradio = False
        else:
            self.has_gradio = True

    def run(self):
        """
        Inicia o dashboard.
        """
        if not self.has_gradio:
            logger.warning(
                "Dashboard requer Gradio. Instalando com: pip install gradio"
            )
            self._text_interface()
            return

        # Criar interface Gradio
        with gr.Blocks(
            title="CerBerus.AI - Pipeline Dashboard", theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("# CerBerus.AI - Pipeline Builder Dashboard")

            with gr.Tabs():
                # Aba de listagem de pipelines
                with gr.TabItem("Pipelines"):
                    with gr.Row():
                        with gr.Column():
                            refresh_btn = gr.Button("Atualizar Lista")
                            pipelines_output = gr.Dataframe(
                                headers=[
                                    "ID",
                                    "Nome",
                                    "Status",
                                    "Tamanho (MB)",
                                    "Tokens",
                                    "Categorias",
                                ],
                                datatype=[
                                    "str",
                                    "str",
                                    "str",
                                    "number",
                                    "number",
                                    "str",
                                ],
                                label="Pipelines Disponíveis",
                            )

                            refresh_btn.click(
                                fn=self._list_pipelines_gradio, outputs=pipelines_output
                            )

                    with gr.Row():
                        with gr.Column():
                            pipeline_id_input = gr.Textbox(label="ID do Pipeline")
                            view_details_btn = gr.Button("Ver Detalhes")
                            details_output = gr.JSON(label="Detalhes do Pipeline")

                            view_details_btn.click(
                                fn=self._get_pipeline_details,
                                inputs=pipeline_id_input,
                                outputs=details_output,
                            )

                    with gr.Row():
                        with gr.Column():
                            approve_id_input = gr.Textbox(label="ID do Pipeline")
                            approve_user_input = gr.Textbox(
                                label="ID do Usuário", value="admin"
                            )
                            approve_comment = gr.Textbox(label="Comentário")

                            with gr.Row():
                                approve_btn = gr.Button("Aprovar Pipeline")
                                reject_btn = gr.Button("Rejeitar Pipeline")

                            approval_output = gr.JSON(label="Resultado da Aprovação")

                            approve_btn.click(
                                fn=lambda pid, uid, comment: self._approve_pipeline(
                                    pid, uid, True, comment
                                ),
                                inputs=[
                                    approve_id_input,
                                    approve_user_input,
                                    approve_comment,
                                ],
                                outputs=approval_output,
                            )

                            reject_btn.click(
                                fn=lambda pid, uid, comment: self._approve_pipeline(
                                    pid, uid, False, comment
                                ),
                                inputs=[
                                    approve_id_input,
                                    approve_user_input,
                                    approve_comment,
                                ],
                                outputs=approval_output,
                            )

                # Aba de criação de pipeline
                with gr.TabItem("Criar Pipeline"):
                    with gr.Row():
                        with gr.Column():
                            pipeline_name = gr.Textbox(label="Nome do Pipeline")
                            pipeline_desc = gr.Textbox(label="Descrição", lines=3)

                            with gr.Row():
                                category_select = gr.Dropdown(
                                    choices=[c.value for c in ContentCategory],
                                    label="Categorias",
                                    multiselect=True,
                                )

                            sources_json = gr.Code(
                                label="Fontes de Dados (JSON)",
                                language="json",
                                value=json.dumps(
                                    [
                                        {
                                            "name": "Exemplo Web",
                                            "source_type": "web",
                                            "location": "https://example.com",
                                            "description": "Página de exemplo",
                                        }
                                    ],
                                    indent=2,
                                ),
                            )

                            create_btn = gr.Button("Criar Pipeline")
                            create_output = gr.JSON(label="Resultado da Criação")

                            create_btn.click(
                                fn=self._create_pipeline_gradio,
                                inputs=[
                                    pipeline_name,
                                    pipeline_desc,
                                    category_select,
                                    sources_json,
                                ],
                                outputs=create_output,
                            )

                # Aba de processamento de pipeline
                with gr.TabItem("Processar Pipeline"):
                    with gr.Row():
                        with gr.Column():
                            process_id_input = gr.Textbox(label="ID do Pipeline")
                            process_btn = gr.Button("Processar Pipeline")
                            process_output = gr.JSON(label="Resultado do Processamento")

                            process_btn.click(
                                fn=self._process_pipeline,
                                inputs=process_id_input,
                                outputs=process_output,
                            )

                    with gr.Row():
                        gr.Markdown(
                            """
                        ### Sobre o Processamento
                        
                        O processamento de um pipeline envolve:
                        
                        1. **Coleta**: Obter dados das fontes definidas
                        2. **Análise**: Processar e analisar o conteúdo coletado
                        3. **Indexação**: Preparar dados para busca e recuperação
                        
                        O tempo de processamento varia de acordo com o número e tipo das fontes.
                        """
                        )

                # Aba de exploração de pipeline
                with gr.TabItem("Explorar Conteúdo"):
                    with gr.Row():
                        with gr.Column():
                            explore_id_input = gr.Textbox(label="ID do Pipeline")
                            explore_btn = gr.Button("Explorar Conteúdo")
                            explore_output = gr.Markdown(label="Conteúdo do Pipeline")

                            explore_btn.click(
                                fn=self._explore_pipeline_content,
                                inputs=explore_id_input,
                                outputs=explore_output,
                            )

                # Aba de análise avançada
                with gr.TabItem("Análise Avançada"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_id_input = gr.Textbox(label="ID do Pipeline")
                            analysis_btn = gr.Button("Analisar Pipeline")

                    with gr.Row():
                        analysis_plot = gr.Plot(label="Visualização de Resultados")

                    analysis_btn.click(
                        fn=self._visualize_analysis_results,
                        inputs=analysis_id_input,
                        outputs=analysis_plot,
                    )

                # Aba de estatísticas
                with gr.TabItem("Estatísticas"):
                    with gr.Row():
                        with gr.Column():
                            stats_btn = gr.Button("Calcular Estatísticas")
                            stats_output = gr.Plot(label="Estatísticas de Pipelines")

                            stats_btn.click(
                                fn=self._generate_pipeline_stats, outputs=stats_output
                            )

                # Aba de ajuda
                with gr.TabItem("Ajuda"):
                    with gr.Row():
                        gr.Markdown(
                            """
                        ## Ajuda do Pipeline Builder Dashboard
                        
                        ### Visão Geral
                        
                        Este dashboard permite gerenciar e visualizar os pipelines de ingestão de dados.
                        
                        ### Funções Principais
                        
                        1. **Pipelines**: Lista todos os pipelines disponíveis e permite ver detalhes.
                        2. **Criar Pipeline**: Interface para criar novos pipelines com diferentes fontes.
                        3. **Processar Pipeline**: Executa a coleta e análise de dados de um pipeline.
                        4. **Explorar Conteúdo**: Visualiza o conteúdo coletado em um pipeline.
                        5. **Análise Avançada**: Ferramentas de visualização para resultados de análise.
                        6. **Estatísticas**: Gráficos e estatísticas sobre todos os pipelines.
                        
                        ### Requisitos
                        
                        Para utilizar todas as funcionalidades de visualização, instale:
                        ```
                        pip install gradio matplotlib networkx numpy scikit-learn
                        ```
                        
                        ### Suporte
                        
                        Para mais informações, consulte a documentação ou entre em contato com o suporte.
                        """
                        )

            # Carregar pipelines automaticamente ao iniciar
            interface.load(fn=self._list_pipelines_gradio, outputs=pipelines_output)

        # Iniciar servidor Gradio
        interface.launch(server_name="0.0.0.0", server_port=self.port, share=False)

    def _text_interface(self):
        """
        Interface de texto simples quando Gradio não está disponível.
        """
        print("\n=== CerBerus.AI - Pipeline Builder Dashboard ===")

        while True:
            print("\nOpções:")
            print("1. Listar pipelines")
            print("2. Ver detalhes de um pipeline")
            print("3. Aprovar/rejeitar pipeline")
            print("4. Sair")

            choice = input("\nEscolha uma opção: ")

            if choice == "1":
                pipelines = self.pipeline_builder.list_pipelines()
                print("\nPipelines disponíveis:")
                print("-" * 80)
                print(
                    f"{'ID':<36} | {'Nome':<30} | {'Status':<10} | {'Tamanho (MB)':<12} | {'Tokens':<10}"
                )
                print("-" * 80)
                for p in pipelines:
                    size_mb = p.get("size_bytes", 0) / (1024 * 1024)
                    print(
                        f"{p.get('pipeline_id', 'N/A'):<36} | {p.get('name', 'N/A'):<30} | {p.get('status', 'N/A'):<10} | {size_mb:<12.2f} | {p.get('estimated_tokens', 0):<10}"
                    )

            elif choice == "2":
                pipeline_id = input("Digite o ID do pipeline: ")
                try:
                    details = self.pipeline_builder.get_pipeline(pipeline_id)
                    print("\nDetalhes do pipeline:")
                    print(json.dumps(details, indent=2))
                except Exception as e:
                    print(f"Erro ao buscar detalhes: {e}")

            elif choice == "3":
                pipeline_id = input("Digite o ID do pipeline: ")
                user_id = input("Digite o ID do usuário: ")
                approve = input("Aprovar pipeline? (s/n): ").lower() == "s"
                comment = input("Comentário (opcional): ")

                try:
                    result = self.pipeline_builder.approve_pipeline(
                        pipeline_id=pipeline_id,
                        user_id=user_id,
                        approved=approve,
                        comment=comment,
                    )
                    print("\nResultado:")
                    print(
                        f"Pipeline {pipeline_id} {'aprovado' if approve else 'rejeitado'} com sucesso."
                    )
                except Exception as e:
                    print(f"Erro na aprovação: {e}")

            elif choice == "4":
                print("Saindo do dashboard...")
                break

            else:
                print("Opção inválida!")

    def _list_pipelines_gradio(self):
        """
        Lista pipelines para o Gradio.

        Returns:
            Lista de pipelines formatada para Gradio.
        """
        pipelines = self.pipeline_builder.list_pipelines()
        rows = []

        for p in pipelines:
            size_mb = p.get("size_bytes", 0) / (1024 * 1024)
            categories = ", ".join(p.get("categories", []))

            rows.append(
                [
                    p.get("pipeline_id", ""),
                    p.get("name", ""),
                    p.get("status", ""),
                    round(size_mb, 2),
                    p.get("estimated_tokens", 0),
                    categories,
                ]
            )

        return rows

    def _get_pipeline_details(self, pipeline_id: str):
        """
        Obtém detalhes de um pipeline.

        Args:
            pipeline_id: ID do pipeline.

        Returns:
            Detalhes do pipeline.
        """
        try:
            return self.pipeline_builder.get_pipeline(pipeline_id)
        except Exception as e:
            logger.error(f"Erro ao obter detalhes do pipeline {pipeline_id}: {e}")
            return {"error": str(e)}

    def _approve_pipeline(
        self, pipeline_id: str, user_id: str, approved: bool, comment: str
    ):
        """
        Aprova ou rejeita um pipeline.

        Args:
            pipeline_id: ID do pipeline.
            user_id: ID do usuário.
            approved: Se o pipeline foi aprovado.
            comment: Comentário opcional.

        Returns:
            Resultado da aprovação.
        """
        try:
            result = self.pipeline_builder.approve_pipeline(
                pipeline_id=pipeline_id,
                user_id=user_id,
                approved=approved,
                comment=comment,
            )

            return {
                "success": True,
                "pipeline_id": pipeline_id,
                "approved": approved,
                "user_id": user_id,
                "timestamp": result.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(
                f"Erro ao {'aprovar' if approved else 'rejeitar'} pipeline {pipeline_id}: {e}"
            )
            return {"error": str(e)}

    def _create_pipeline_gradio(
        self, name: str, description: str, categories: List[str], sources_json: str
    ):
        """
        Cria um pipeline a partir dos dados da interface Gradio.

        Args:
            name: Nome do pipeline.
            description: Descrição do pipeline.
            categories: Lista de categorias.
            sources_json: Fontes de dados em formato JSON.

        Returns:
            Resultado da criação.
        """
        try:
            # Converter categorias para enum
            category_enums = [ContentCategory(cat) for cat in categories]

            # Converter JSON de fontes para objetos DataSource
            sources_data = json.loads(sources_json)
            sources = []

            for src in sources_data:
                sources.append(
                    DataSource(
                        name=src["name"],
                        source_type=src["source_type"],
                        location=src["location"],
                        description=src.get("description", ""),
                        metadata=src.get("metadata", {}),
                    )
                )

            # Criar pipeline
            pipeline = self.pipeline_builder.build_pipeline(
                sources=sources,
                name=name,
                description=description,
                categories=category_enums,
            )

            return {
                "success": True,
                "pipeline_id": pipeline.id,
                "name": pipeline.name,
                "status": pipeline.status.value,
                "size_mb": pipeline.size / (1024 * 1024),
                "estimated_tokens": pipeline.estimated_tokens,
            }

        except Exception as e:
            logger.error(f"Erro ao criar pipeline: {e}")
            return {"error": str(e)}

    def _process_pipeline(self, pipeline_id: str):
        """
        Processa um pipeline diretamente da interface.

        Args:
            pipeline_id: ID do pipeline a ser processado.

        Returns:
            JSON com o resultado do processamento.
        """
        try:
            # Obter pipeline
            pipeline = self.pipeline_builder.get_pipeline(pipeline_id)
            if not pipeline:
                return {"error": "Pipeline não encontrado"}

            # Iniciar processamento
            result = self.pipeline_builder.process_pipeline(pipeline_id)

            # Formatar resultados
            if result:
                return {
                    "status": "sucesso",
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline.name,
                    "chunks_coletados": len(result.get("chunks", [])),
                    "resultados_análise": len(result.get("analysis_results", [])),
                    "tempo_processamento": result.get("processing_time"),
                    "status_final": result.get("status"),
                }
            else:
                return {
                    "error": "Falha ao processar pipeline",
                    "pipeline_id": pipeline_id,
                }

        except Exception as e:
            logger.error(f"Erro ao processar pipeline: {e}")
            return {"error": str(e), "pipeline_id": pipeline_id}

    def _explore_pipeline_content(self, pipeline_id: str):
        """
        Explora o conteúdo de um pipeline, mostrando chunks e relações.

        Args:
            pipeline_id: ID do pipeline a ser explorado.

        Returns:
            Markdown formatado com o conteúdo do pipeline.
        """
        try:
            # Obter pipeline
            pipeline = self.pipeline_builder.get_pipeline(pipeline_id)
            if not pipeline:
                return "Pipeline não encontrado."

            # Obter chunks do pipeline
            chunks = self.pipeline_builder.get_pipeline_chunks(pipeline_id)
            if not chunks:
                return "Nenhum conteúdo encontrado neste pipeline."

            # Obter resultados de análise
            analysis_results = self.pipeline_builder.get_pipeline_analysis_results(
                pipeline_id
            )

            # Formatar saída em Markdown
            output = f"## Conteúdo do Pipeline: {pipeline.name}\n\n"
            output += f"**Total de chunks:** {len(chunks)}\n\n"

            # Incluir informações sobre fontes
            sources = self.pipeline_builder.get_pipeline_sources(pipeline_id)
            if sources:
                output += "### Fontes de Dados\n\n"
                for i, source in enumerate(sources):
                    output += f"**{i+1}. {source.name}** ({source.source_type.value})\n"
                    output += f"   - Localização: {source.location}\n"
                    output += f"   - Descrição: {source.description}\n\n"

            # Adicionar visualização de grafo se Gradio suportar
            if hasattr(gr, "Plot"):
                try:
                    import networkx as nx
                    import matplotlib.pyplot as plt
                    import numpy as np
                    from sklearn.manifold import TSNE
                    from sklearn.cluster import KMeans

                    # Criar grafo
                    G = nx.Graph()

                    # Adicionar nós (chunks)
                    for i, chunk in enumerate(
                        chunks[:50]
                    ):  # Limitar a 50 para performance
                        G.add_node(
                            chunk.id, text=chunk.text[:50] + "...", size=chunk.length
                        )

                    # Adicionar arestas (similaridade entre chunks)
                    if analysis_results:
                        for result in analysis_results:
                            if "similarity_scores" in result.metadata:
                                for target_id, score in result.metadata[
                                    "similarity_scores"
                                ].items():
                                    if (
                                        score > 0.7
                                        and G.has_node(target_id)
                                        and G.has_node(result.chunk_id)
                                    ):
                                        G.add_edge(
                                            result.chunk_id, target_id, weight=score
                                        )

                    # Criar visualização
                    plt.figure(figsize=(12, 8))

                    # Configurar layout
                    pos = nx.spring_layout(G)

                    # Desenhar nós e arestas
                    nx.draw_networkx_nodes(
                        G, pos, node_size=100, node_color="skyblue", alpha=0.8
                    )
                    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

                    # Adicionar labels
                    labels = {
                        node: data["text"][:20] + "..."
                        for node, data in G.nodes(data=True)
                    }
                    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

                    plt.title(f"Grafo de Similaridade - {pipeline.name}")
                    plt.axis("off")

                    # Retornar visualização como parte da saída
                    return output
                except ImportError:
                    output += "\n\n*Nota: Instale networkx, matplotlib, numpy e scikit-learn para visualização avançada.*\n\n"

            # Mostrar amostra de chunks
            output += "### Amostra de Conteúdo\n\n"
            for i, chunk in enumerate(chunks[:10]):  # Mostrar apenas os 10 primeiros
                output += f"**Chunk {i+1}:** {chunk.source_location}\n"
                output += f"```\n{chunk.text[:200]}...\n```\n\n"

                # Adicionar informações de análise se disponíveis
                for result in analysis_results:
                    if result.chunk_id == chunk.id:
                        output += "**Análise:**\n"
                        if "keywords" in result.metadata:
                            output += f"- Palavras-chave: {', '.join(result.metadata['keywords'][:5])}\n"
                        if "word_count" in result.metadata:
                            output += f"- Contagem de palavras: {result.metadata['word_count']}\n"
                        if "relevance_score" in result.metadata:
                            output += f"- Pontuação de relevância: {result.metadata['relevance_score']:.2f}\n"
                        if "summary" in result.metadata:
                            output += f"- Resumo: {result.metadata['summary']}\n"
                        output += "\n"

            return output
        except Exception as e:
            logger.error(f"Erro ao explorar pipeline: {e}")
            return f"Erro ao explorar pipeline: {str(e)}"

    def _generate_pipeline_stats(self):
        """
        Gera estatísticas e visualizações sobre os pipelines.

        Returns:
            Objeto de plotagem com estatísticas.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Obter todos os pipelines
            pipelines = self.pipeline_builder.list_pipelines()

            if not pipelines:
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    "Nenhum pipeline encontrado",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                plt.axis("off")
                return fig

            # Coletar dados para visualização
            pipeline_names = []
            chunk_counts = []
            token_counts = []
            source_types = {}

            for pipeline in pipelines:
                pipeline_names.append(pipeline.name)

                # Contar chunks
                chunks = self.pipeline_builder.get_pipeline_chunks(pipeline.id)
                chunk_count = len(chunks) if chunks else 0
                chunk_counts.append(chunk_count)

                # Estimar tokens
                tokens = sum(chunk.length // 4 for chunk in chunks) if chunks else 0
                token_counts.append(tokens)

                # Contar tipos de fontes
                sources = self.pipeline_builder.get_pipeline_sources(pipeline.id)
                for source in sources:
                    source_type = source.source_type.value
                    if source_type in source_types:
                        source_types[source_type] += 1
                    else:
                        source_types[source_type] = 1

            # Criar visualização com múltiplos subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Gráfico 1: Contagem de chunks por pipeline (barras)
            axs[0, 0].bar(pipeline_names, chunk_counts, color="skyblue")
            axs[0, 0].set_title("Chunks por Pipeline")
            axs[0, 0].set_xlabel("Pipeline")
            axs[0, 0].set_ylabel("Número de Chunks")
            axs[0, 0].tick_params(axis="x", rotation=45)

            # Gráfico 2: Estimativa de tokens por pipeline (barras)
            axs[0, 1].bar(pipeline_names, token_counts, color="lightgreen")
            axs[0, 1].set_title("Tokens Estimados por Pipeline")
            axs[0, 1].set_xlabel("Pipeline")
            axs[0, 1].set_ylabel("Número de Tokens")
            axs[0, 1].tick_params(axis="x", rotation=45)

            # Gráfico 3: Distribuição de tipos de fontes (pizza)
            if source_types:
                axs[1, 0].pie(
                    source_types.values(), labels=source_types.keys(), autopct="%1.1f%%"
                )
                axs[1, 0].set_title("Distribuição de Tipos de Fontes")
            else:
                axs[1, 0].text(
                    0.5,
                    0.5,
                    "Nenhuma fonte encontrada",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[1, 0].axis("off")

            # Gráfico 4: Comparação de tamanho total entre pipelines
            if chunk_counts and token_counts:
                indices = np.arange(len(pipeline_names))
                width = 0.35

                # Normalizar dados para tornar o gráfico mais legível
                normalized_chunks = np.array(chunk_counts) / max(chunk_counts) * 100
                normalized_tokens = np.array(token_counts) / max(token_counts) * 100

                axs[1, 1].bar(
                    indices - width / 2, normalized_chunks, width, label="Chunks (%)"
                )
                axs[1, 1].bar(
                    indices + width / 2, normalized_tokens, width, label="Tokens (%)"
                )
                axs[1, 1].set_xticks(indices)
                axs[1, 1].set_xticklabels(pipeline_names)
                axs[1, 1].set_title("Comparação de Tamanho Relativo")
                axs[1, 1].set_xlabel("Pipeline")
                axs[1, 1].set_ylabel("Percentual do Máximo")
                axs[1, 1].legend()
                axs[1, 1].tick_params(axis="x", rotation=45)
            else:
                axs[1, 1].text(
                    0.5,
                    0.5,
                    "Dados insuficientes",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[1, 1].axis("off")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Erro ao gerar estatísticas: {e}")
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Erro ao gerar estatísticas: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            plt.axis("off")
            return fig

    def _visualize_analysis_results(self, pipeline_id: str):
        """
        Gera visualizações detalhadas dos resultados de análise.

        Args:
            pipeline_id: ID do pipeline.

        Returns:
            Objeto de plotagem com visualizações.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.manifold import TSNE

            # Obter pipeline
            pipeline = self.pipeline_builder.get_pipeline(pipeline_id)
            if not pipeline:
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    "Pipeline não encontrado",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                plt.axis("off")
                return fig

            # Obter chunks e resultados de análise
            chunks = self.pipeline_builder.get_pipeline_chunks(pipeline_id)
            analysis_results = self.pipeline_builder.get_pipeline_analysis_results(
                pipeline_id
            )

            if not chunks or not analysis_results:
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    "Nenhum dado de análise encontrado",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                plt.axis("off")
                return fig

            # Criar visualização com múltiplos subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Gráfico 1: Pontuações de relevância (se disponíveis)
            relevance_scores = []
            chunk_ids = []

            for result in analysis_results:
                if "relevance_score" in result.metadata:
                    relevance_scores.append(result.metadata["relevance_score"])

                    # Encontrar o texto do chunk para o label
                    chunk_text = "Desconhecido"
                    for chunk in chunks:
                        if chunk.id == result.chunk_id:
                            chunk_text = chunk.text[:20] + "..."
                            break

                    chunk_ids.append(chunk_text)

            if relevance_scores and chunk_ids:
                # Ordenar por relevância
                sorted_indices = np.argsort(relevance_scores)[::-1][:10]  # Top 10
                top_scores = [relevance_scores[i] for i in sorted_indices]
                top_chunks = [chunk_ids[i] for i in sorted_indices]

                axs[0, 0].barh(top_chunks, top_scores, color="skyblue")
                axs[0, 0].set_title("Top 10 Chunks por Relevância")
                axs[0, 0].set_xlabel("Pontuação de Relevância")
            else:
                axs[0, 0].text(
                    0.5,
                    0.5,
                    "Dados de relevância não disponíveis",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[0, 0].axis("off")

            # Gráfico 2: Distribuição de comprimento dos chunks
            if chunks:
                chunk_lengths = [chunk.length for chunk in chunks]
                axs[0, 1].hist(chunk_lengths, bins=20, color="lightgreen", alpha=0.7)
                axs[0, 1].set_title("Distribuição de Tamanho dos Chunks")
                axs[0, 1].set_xlabel("Comprimento (caracteres)")
                axs[0, 1].set_ylabel("Número de Chunks")
            else:
                axs[0, 1].text(
                    0.5,
                    0.5,
                    "Nenhum chunk encontrado",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[0, 1].axis("off")

            # Gráfico 3: Visualização de embeddings em 2D (se disponíveis)
            embeddings = []
            labels = []

            for result in analysis_results:
                if "embedding" in result.metadata:
                    embeddings.append(result.metadata["embedding"])

                    # Encontrar o texto do chunk para o label
                    for chunk in chunks:
                        if chunk.id == result.chunk_id:
                            labels.append(chunk.text[:10] + "...")
                            break
                    else:
                        labels.append("Desconhecido")

            if embeddings and len(embeddings) > 5:
                try:
                    # Reduzir dimensionalidade para visualização 2D
                    embeddings_array = np.array(embeddings)
                    tsne = TSNE(n_components=2, random_state=42)
                    embeddings_2d = tsne.fit_transform(embeddings_array)

                    # Criar scatter plot
                    axs[1, 0].scatter(
                        embeddings_2d[:, 0],
                        embeddings_2d[:, 1],
                        alpha=0.7,
                        s=50,
                        c=range(len(embeddings_2d)),
                        cmap="viridis",
                    )
                    axs[1, 0].set_title("Visualização 2D de Embeddings")

                    # Adicionar alguns labels (não todos para evitar poluição visual)
                    for i in range(min(10, len(labels))):
                        axs[1, 0].annotate(
                            labels[i],
                            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                            fontsize=8,
                            alpha=0.7,
                        )
                except Exception as e:
                    axs[1, 0].text(
                        0.5,
                        0.5,
                        f"Erro ao visualizar embeddings: {str(e)}",
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    axs[1, 0].axis("off")
            else:
                axs[1, 0].text(
                    0.5,
                    0.5,
                    "Embeddings não disponíveis ou insuficientes",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[1, 0].axis("off")

            # Gráfico 4: Nuvem de palavras-chave
            keywords_counter = {}

            for result in analysis_results:
                if "keywords" in result.metadata:
                    for keyword in result.metadata["keywords"]:
                        if keyword in keywords_counter:
                            keywords_counter[keyword] += 1
                        else:
                            keywords_counter[keyword] = 1

            if keywords_counter:
                # Mostrar as palavras-chave mais comuns
                top_keywords = sorted(
                    keywords_counter.items(), key=lambda x: x[1], reverse=True
                )[:20]
                keywords = [k for k, _ in top_keywords]
                counts = [c for _, c in top_keywords]

                axs[1, 1].barh(keywords, counts, color="coral")
                axs[1, 1].set_title("Palavras-chave Mais Comuns")
                axs[1, 1].set_xlabel("Frequência")
            else:
                axs[1, 1].text(
                    0.5,
                    0.5,
                    "Dados de palavras-chave não disponíveis",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                axs[1, 1].axis("off")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Erro ao visualizar resultados: {e}")
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Erro ao visualizar resultados: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            plt.axis("off")
            return fig
