"""课程脉络与进度追踪：58 章 → 10 个阶段 → 可持久化的学习进度。

第 2 章要回答"课程分几个阶段、先后关系如何"。本章把它变成**数据结构**：
每章带阶段、标题、关键词、前置依赖，脚本能告诉你"下一章该学什么"、"哪些前置没读完"。

运行::

    uv run python -m chapters.ch02_course_roadmap.roadmap
    uv run python -m chapters.ch02_course_roadmap.roadmap --done 4 --done 5 --next
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_course.logging import get_console

console = get_console()

PROGRESS_FILE = Path(__file__).resolve().parent / "workspace" / "progress.json"


@dataclass(frozen=True, slots=True)
class Chapter:
    """一章的元数据。

    Attributes:
        index: 章节序号（1–58）。
        slug: 本仓库中的目录名（chNN_xxx）。
        title: 中文标题。
        stage: 所属阶段名。
        keywords: 该章核心概念，便于检索与复习。
        requires: 前置章节序号。
    """

    index: int
    slug: str
    title: str
    stage: str
    keywords: tuple[str, ...] = ()
    requires: tuple[int, ...] = ()


@dataclass(slots=True)
class Progress:
    """学习进度（可 JSON 持久化）。"""

    done: set[int] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 文本。"""
        return json.dumps({"done": sorted(self.done), "notes": self.notes}, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path = PROGRESS_FILE) -> Progress:
        """从磁盘加载；文件不存在时返回空进度。"""
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(done=set(raw.get("done", [])), notes=dict(raw.get("notes", {})))

    def save(self, path: Path = PROGRESS_FILE) -> Path:
        """写入磁盘（自动建目录）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


S1 = "S1 开篇与学习地图"
S2 = "S2 Tool 与 MCP"
S3 = "S3 RAG 基础"
S4 = "S4 Memory 与结构化输出"
S5 = "S5 Prompt / LCEL"
S6 = "S6 服务化与工程化"
S7 = "S7 LangGraph 与多 Agent"
S8 = "S8 检索与基础设施"
S9 = "S9 原理、简历与面试"
S10 = "S10 企业级知识库实战"

CHAPTERS: tuple[Chapter, ...] = (
    Chapter(1, "ch01_transition_story", "转型经验与能力差距分析", S1, ("能力模型", "学习计划"), ()),
    Chapter(2, "ch02_course_roadmap", "课程脉络与进度追踪", S1, ("学习地图", "依赖拓扑"), (1,)),
    Chapter(3, "ch03_what_to_learn", "AI Agent 开发要学什么", S1, ("技能树", "技术选型"), (2,)),
    Chapter(4, "ch04_tool_basics", "从 Tool 开始：让模型调工具", S2, ("bind_tools", "ToolMessage"), (3,)),
    Chapter(5, "ch05_mini_cursor", "mini cursor：让模型执行命令", S2, ("沙箱", "命令白名单"), (4,)),
    Chapter(6, "ch06_mcp_server", "MCP：可跨进程调用的 Tool", S2, ("JSON-RPC", "stdio"), (5,)),
    Chapter(7, "ch07_reuse_mcp", "复用别人的 MCP Server", S2, ("MCP 客户端", "工具生态"), (6,)),
    Chapter(8, "ch08_rag_vector", "RAG：向量化与语义搜索", S3, ("Embedding", "余弦相似度"), (4,)),
    Chapter(9, "ch09_loader_splitter", "loader 与 splitter", S3, ("文档加载", "切分"), (8,)),
    Chapter(10, "ch10_splitter_deep", "切分策略与质量评估", S3, ("token 预算", "重叠"), (9,)),
    Chapter(11, "ch11_milvus", "向量数据库 Milvus", S3, ("HNSW", "过滤检索"), (8,)),
    Chapter(12, "ch12_milvus_rag", "Milvus + RAG 实战", S3, ("批量写入", "元数据过滤"), (11,)),
    Chapter(13, "ch13_memory_strategy", "Memory 三大策略", S4, ("截断", "总结", "检索"), (8,)),
    Chapter(14, "ch14_structured_output", "结构化输出：parser 还是 tool", S4, ("Pydantic", "校验重试"), (4,)),
    Chapter(15, "ch15_output_parser_practice", "Output Parser 实战", S4, ("流式解析", "智能录入"), (14,)),
    Chapter(16, "ch16_prompt_template", "Prompt Template 组件化", S5, ("FewShot", "ExampleSelector"), (4,)),
    Chapter(17, "ch17_runnable", "Runnable 与 LCEL", S5, ("管道", "RunnableParallel"), (16,)),
    Chapter(18, "ch18_lcel_practice", "LCEL 组装实战", S5, ("分支", "fallback"), (17,)),
    Chapter(19, "ch19_langchain_summary", "第一阶段总结与自测", S5, ("能力清单", "复盘"), (18,)),
    Chapter(20, "ch20_sse_stream", "SSE 流式接口", S6, ("FastAPI", "StreamingResponse"), (17,)),
    Chapter(21, "ch21_cron_tool_part1", "定时任务（上）", S6, ("调度器", "cron 解析"), (4,)),
    Chapter(22, "ch22_cron_tool_part2", "定时任务（下）", S6, ("任务持久化", "幂等"), (21,)),
    Chapter(23, "ch23_asr_tts", "语音交互：ASR + TTS", S6, ("流式音频", "协议抽象"), (20,)),
    Chapter(24, "ch24_agui_protocol", "AGUI 流式组件渲染", S6, ("事件流", "组件协议"), (20,)),
    Chapter(25, "ch25_langgraph", "LangGraph 与多 Agent", S7, ("StateGraph", "checkpoint"), (14,)),
    Chapter(26, "ch26_agentic_rag", "Agentic RAG 闭环", S7, ("查询重写", "多跳"), (25,)),
    Chapter(27, "ch27_docker_compose", "Docker Compose 部署", S8, ("多阶段构建", "健康检查"), ()),
    Chapter(28, "ch28_elasticsearch", "Elasticsearch 与 BM25", S8, ("倒排索引", "分词"), ()),
    Chapter(29, "ch29_hybrid_search", "混合检索与重排", S8, ("多路召回", "RRF"), (28, 11)),
    Chapter(30, "ch30_neo4j_graphrag", "Neo4j 与 Graph RAG", S8, ("知识图谱", "Cypher"), ()),
    Chapter(31, "ch31_langsmith", "LangSmith 观测与评估", S8, ("trace", "评测集"), (12,)),
    Chapter(32, "ch32_deepagents", "DeepAgents 中间件", S8, ("middleware", "上下文压缩"), (25,)),
    Chapter(33, "ch33_deep_research", "多 Agent 深度调研", S7, ("supervisor", "子 Agent"), (32,)),
    Chapter(34, "ch34_postgresql", "PostgreSQL 与 pgvector", S8, ("SQLAlchemy", "向量扩展"), ()),
    Chapter(35, "ch35_redis_memory", "Redis 短期记忆", S8, ("会话存储", "TTL"), ()),
    Chapter(36, "ch36_mem0", "Mem0 长期记忆", S8, ("分层记忆", "多路召回"), (35,)),
    Chapter(37, "ch37_fastapi_backend", "FastAPI 企业级后端", S6, ("依赖注入", "分层架构"), (20,)),
    Chapter(38, "ch38_object_storage", "对象存储 MinIO / OSS", S8, ("预签名", "分片上传"), ()),
    Chapter(39, "ch39_multimodal_upload", "多模态与前端直传", S8, ("多模态输入", "直传"), (38,)),
    Chapter(40, "ch40_rabbitmq", "RabbitMQ 异步处理", S8, ("交换机", "消费确认"), ()),
    Chapter(41, "ch41_langfuse", "LangFuse 内网观测", S8, ("trace", "评分"), (31,)),
    Chapter(42, "ch42_transformer", "Transformer 原理", S9, ("注意力", "KV Cache"), ()),
    Chapter(43, "ch43_training_inference", "训练与推理全流程", S9, ("预训练", "量化"), (42,)),
    Chapter(44, "ch44_resume_library", "简历参考库（一）", S9, ("项目描述", "量化成果"), ()),
    Chapter(45, "ch45_resume_library_2", "简历参考库（二）", S9, ("STAR", "避坑"), (44,)),
    Chapter(46, "ch46_interview_rag", "RAG 面试题精讲", S9, ("高频题", "答题结构"), (12, 29)),
    Chapter(47, "ch47_resume_rag", "简历 RAG 专场", S9, ("岗位匹配", "亮点提炼"), (44,)),
    Chapter(48, "ch48_kb_project_intro", "企业级知识库：架构梳理", S10, ("多模态 RAG", "分层"), (29,)),
    Chapter(49, "ch49_kb_database_design", "文档模块数据库设计", S10, ("PG + Mongo", "状态机"), (34,)),
    Chapter(50, "ch50_document_parsing", "PDF/Office 解析为 Markdown", S10, ("解析管线", "表格还原"), (49,)),
    Chapter(51, "ch51_async_rag_pipeline", "消息队列异步 RAG", S10, ("异步流水线", "幂等"), (40, 50)),
    Chapter(52, "ch52_fulltext_search", "全文检索链路", S10, ("高亮", "排序"), (28,)),
    Chapter(53, "ch53_neo4j_extraction", "图谱实体抽取", S10, ("实体关系", "入图"), (30,)),
    Chapter(54, "ch54_document_audit", "文档审核状态机", S10, ("状态流转", "审计"), (49,)),
    Chapter(55, "ch55_auth_flow", "用户鉴权流程", S10, ("JWT", "刷新令牌"), (37,)),
    Chapter(56, "ch56_user_module", "用户模块完善", S10, ("资料", "配额"), (55,)),
    Chapter(57, "ch57_rbac", "RBAC 权限系统", S10, ("角色", "权限校验"), (56,)),
    Chapter(
        58, "ch58_highlight_graph_search", "高亮检索 + 图谱检索", S10, ("混合检索", "图谱召回"), (52, 53)
    ),
)

BY_INDEX: dict[int, Chapter] = {chapter.index: chapter for chapter in CHAPTERS}


def next_chapter(progress: Progress) -> Chapter | None:
    """推荐下一章：优先「前置已全部完成」且未完成的最小序号。"""
    for chapter in CHAPTERS:
        if chapter.index in progress.done:
            continue
        if all(req in progress.done for req in chapter.requires):
            return chapter
    return None


def blocked(progress: Progress) -> list[tuple[Chapter, tuple[int, ...]]]:
    """找出「想学但前置没读完」的章节。"""
    result = []
    for chapter in CHAPTERS:
        if chapter.index in progress.done or not chapter.requires:
            continue
        missing = tuple(req for req in chapter.requires if req not in progress.done)
        if missing:
            result.append((chapter, missing))
    return result


def stage_summary(progress: Progress) -> dict[str, tuple[int, int]]:
    """统计每个阶段「已完成 / 总数」。"""
    summary: dict[str, list[int]] = {}
    for chapter in CHAPTERS:
        done, total = summary.setdefault(chapter.stage, [0, 0])
        summary[chapter.stage] = [done + int(chapter.index in progress.done), total + 1]
    return {stage: (value[0], value[1]) for stage, value in summary.items()}


def main() -> None:
    """命令行入口：查看脉络、记录进度、获取下一步推荐。"""
    parser = argparse.ArgumentParser(description="课程脉络与进度追踪")
    parser.add_argument("--done", type=int, action="append", default=[], help="标记已完成的章节号，可重复")
    parser.add_argument("--next", action="store_true", help="只输出下一章推荐")
    parser.add_argument("--reset", action="store_true", help="清空进度")
    args = parser.parse_args()

    progress = Progress() if args.reset else Progress.load()
    progress.done.update(args.done)

    if args.next:
        recommendation = next_chapter(progress)
        if recommendation is None:
            console.print("[green]全部章节已完成[/green]")
        else:
            console.print(f"[bold]下一章[/bold] {recommendation.index} · {recommendation.title}")
            console.print(f"  目录：chapters/{recommendation.slug}")
            console.print(f"  关键词：{'、'.join(recommendation.keywords)}")
        Progress().save() if args.reset else progress.save()
        return

    console.print("[bold]阶段进度[/bold]")
    for stage, (done, total) in stage_summary(progress).items():
        percent = done / total * 100
        bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
        console.print(f"  {stage:<28} {bar} {done}/{total}")

    console.print(f"\n[bold]已完成 {len(progress.done)}/{len(CHAPTERS)} 章[/bold]")
    for chapter, missing in blocked(progress):
        titles = "、".join(str(BY_INDEX[idx].index) for idx in missing)
        console.print(f"  ⚠ {chapter.index} {chapter.title} 缺少前置：{titles}")

    path = progress.save()
    console.print(f"\n[dim]进度已保存到 {path}[/dim]")


if __name__ == "__main__":
    main()
