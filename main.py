import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# 加载 .env 文件中的环境变量，避免在代码中硬编码 API Key
load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    raise ValueError("未检测到有效的 OPENAI_API_KEY，请检查 .env 文件！")

def create_competitive_analysis_crew(competitor_name, industry):
    """
    创建并组装竞品分析多智能体团队
    """
    # 1. 初始化工具与大模型
    search_tool = DuckDuckGoSearchRun()
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.4)

    # 2. 定义智能体角色 (Agents)
    intelligence_scout = Agent(
        role='资深竞品情报侦察员',
        goal=f'全网追踪并收集目标竞品 {competitor_name} 在 {industry} 领域的最新动态、产品发布、新闻和市场动作。',
        backstory='你是一名顶尖的商业间谍和数据矿工，擅长从海量互联网信息中敏锐地捕捉到竞争对手的蛛丝马迹。',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

    market_analyst = Agent(
        role='首席市场分析师',
        goal=f'基于侦察员收集的情报，深度剖析 {competitor_name} 的核心优势、潜在弱点以及对行业格局的影响。',
        backstory='你拥有常春藤名校MBA学位，曾在顶级咨询公司工作。你擅长透过现象看本质，总结对手的商业模式。',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    strategy_consultant = Agent(
        role='己方商业战略顾问',
        goal='基于市场分析师的深度报告，为我们公司提供 3 条高度可执行的应对策略和突围建议。',
        backstory='你是久经沙场的企业战略家。你的建议总是直击要害，能够帮助产品团队立刻采取行动。',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 3. 定义任务流水线 (Tasks)
    scout_task = Task(
        description=f'使用搜索工具，查找关于竞品 "{competitor_name}" 在过去一个月内的最新新闻或市场争议。汇总成原始情报清单。',
        expected_output='一份包含时间节点、事件摘要和数据来源的竞品动态列表。',
        agent=intelligence_scout
    )

    analysis_task = Task(
        description=f'审阅情报清单。使用 SWOT 分析法，评估 "{competitor_name}" 的最新动作对 {industry} 市场的冲击。',
        expected_output='一份结构化的市场分析报告，包含 SWOT 分析矩阵和对手战略意图推测。',
        agent=market_analyst
    )

    strategy_task = Task(
        description=f'审阅市场分析报告。假设我们是 "{competitor_name}" 的直接竞争对手，提出 3 个具体的应对策略。',
        expected_output='一份包含 3 条核心行动指南的战略备忘录。',
        agent=strategy_consultant
    )

    # 4. 组装 Crew
    return Crew(
        agents=[intelligence_scout, market_analyst, strategy_consultant],
        tasks=[scout_task, analysis_task, strategy_task],
        process=Process.sequential, 
        verbose=True
    )

if __name__ == "__main__":
    print("="*50)
    print("🚀 欢迎使用多智能体竞品自动化分析平台 MVP")
    print("="*50)
    
    target_competitor = input("请输入目标竞品名称 (例如: 特斯拉): ") or "特斯拉"
    target_industry = input("请输入所属行业领域 (例如: 新能源汽车): ") or "新能源汽车"
    
    analysis_crew = create_competitive_analysis_crew(target_competitor, target_industry)
    result = analysis_crew.kickoff()
    
    print("\n" + "="*50)
    print("🏆 最终战略报告生成完毕：")
    print("="*50)
    print(result)
