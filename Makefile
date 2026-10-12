# 项目统一入口：所有校验命令都从这里走，避免"我本地是好的"。
# 用法：make check / make test / make smoke / make fix / make coverage
.PHONY: help install check lint format type test coverage fix smoke reproducible experiments output

PY ?= python3
RUFF ?= ruff
MYPY ?= mypy
PYTEST ?= pytest

help:
	@echo "make install   安装依赖（含 dev + rag + web + graph + storage extras）"
	@echo "make check     执行 CI 的全部关卡：lint + format + type + test + smoke"
	@echo "make lint      ruff check"
	@echo "make format    ruff format --check（只校验，不改文件）"
	@echo "make type      mypy --strict"
	@echo "make test      pytest（离线、确定性）"
	@echo "make coverage  覆盖率：pytest + 58 章实验合并（只跑 pytest 会漏掉 1.4 万行 experiments.py）"
	@echo "make smoke     58 章全部入口实跑一遍（离线）"
	@echo "make experiments  58 章动手实验全跑并断言现象"
	@echo "make reproducible  每个入口跑两遍（不同哈希种子），输出必须一致"
	@echo "make output    核对 README 的 run 输出块 == 真跑一遍的输出"
	@echo "make fix       自动修复：ruff check --fix + ruff format"

install:
	$(PY) -m pip install -e ".[dev,rag,web,graph,storage]"

lint:
	$(RUFF) check .

format:
	$(RUFF) format --check .

type:
	$(MYPY) src chapters tests tools site/tools

test:
	$(PYTEST)

coverage:
	# 两遍合并：pytest 跑测试，run_experiments 跑 58 章实验（否则 experiments.py 全算 0%）。
	$(PYTEST) --cov=chapters --cov=src --cov-report=term-missing
	$(PY) tools/run_experiments.py --cov
	$(PY) -m coverage combine
	$(PY) -m coverage report --show-missing

fix:
	$(RUFF) check --fix .
	$(RUFF) format .

smoke:
	$(PY) tools/smoke_chapters.py

experiments:
	$(PY) tools/run_experiments.py

reproducible:
	$(PY) tools/check_reproducible.py

output:
	$(PY) tools/check_readme_output.py

check: lint format type test smoke experiments reproducible output
