---

## RISCOS E MITIGAÇÕES

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| llama_cpp GLIBC não resolvível | Alta | Alto | Fase 1 resolve com providers condicionais; GGUF vira provider opcional |
| Pydantic v2 causa mais quebras | Média | Alto | Fase 1 alinha requirements; migração gradual com `model_dump()` |
| Código órfão é necessário e não percebemos | Média | Médio | Fase 2 integra módulos em vez de arquivar; mover para `legacy/` nunca deleta |
| Integração com Èter quebra o Èter | Alta | Crítico | F4 usa event bus desacoplado; Èter não importa PipelineBuilder |
| Performance degrada com refatoração | Média | Médio | Benchmarks em cada milestone; manter paridade de performance |
| Dependências externas pesadas quebram o sistema | Média | Alto | Fase 1 implementa NLP Engine próprio (SEM NLTK pesado). Providers como dependência opcional |
| Vulnerabilidades de security em código legado | Alta | Crítico | Fase 3 implementa segurança end-to-end: sandbox, whitelist, path traversal blocking, rate limiting |

---

## PRINCÍPIOS DE EXECUÇÃO

1. **Sem muletas:** Nenhum `pass`, stub, placeholder, forward declaration como solução. Implementar TUDO de verdade.
2. **Estudar antes de mexer:** Cada módulo quebrado é analisado, debugado e entendido antes de ser reescrito. Não simplificar por simplificar.
3. **Integrar, não arquivar:** Se módulos devem trabalhar juntos, fazer eles funcionarem juntos. Código com função arquitetural é integrado, não movido para `legacy/`.
4. **Substituir implementação, não arquitetura:** Código lento, inseguro, legacy = reescrito com implementação melhor. A função arquitetural é mantida.
5. **Motor próprio para gargalos:** Se performance é problema, construir motor/processador específico — não improvisar.
6. **Segurança por design:** SQL injection, command injection, path traversal, pickle, XSS — nenhum desses é tolerado. Implementar desde o início.
7. **Èter não é importado pelo PipelineBuilder:** Direção de dependência rigorosa. PipelineBuilder emite eventos; Èter consome.
8. **Plug-and-play é obrigatório:** Nenhum modelo, coletor, analyzer ou processor é hardcoded. Registry + ABCs garantem extensibilidade.
9. **NeuroLex (.vqc) é território proibido:** Qualquer referência a `neurolex`, `vqc`, `vpc` no PipelineBuilder é bloqueada.
10. **Funciona a cada commit:** Nenhum commit deve quebrar o sistema. Feature flags + conditional imports garantem isso.

---

## COMANDOS ÚTEIS DURANTE A EXECUÇÃO

```bash
# Verificação de saúde do pacote
python -c "from cerberus_api.pipeline_builder import PipelineBuilder; print('OK')"

# Lint
ruff check cerberus_api/pipeline_builder/

# Type check
mypy cerberus_api/pipeline_builder/

# Testes
pytest tests/ -v --tb=short

# Verificar imports quebrados
python -c "
import pkgutil, importlib
import cerberus_api.pipeline_builder
for _, modname, ispkg in pkgutil.walk_packages(cerberus_api.pipeline_builder.__path__, cerberus_api.pipeline_builder.__name__ + '.'):
    try:
        importlib.import_module(modname)
    except Exception as e:
        print(f'BROKEN: {modname} -> {e}')
"

# Benchmark simples
python -m cerberus_api.pipeline_builder.cli create --source-type WEB --query "test" --max-pages 10

# Verificar GLIBC
ldd --version | head -1

# Verificar dependências
pip check
```

---

## NOTA FINAL

Este plano é um documento vivo. Cada fase é executada, testada e validada antes de prosseguir. Se durante a execução descobrirmos que uma abordagem não funciona, estudamos, debugamos, e implementamos a alternativa correta — sem simplificações que removam funcionalidade.

O resultado final não é um "sistema que funciona por enquanto". É um sistema robusto, seguro, performático, e correto — porque cada parte foi construída com engenharia de verdade.
