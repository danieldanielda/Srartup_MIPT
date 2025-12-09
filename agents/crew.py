# src/my_project/crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from src.tools.sonar_tool import SonarSearchTool
from src.settings.config import CrewSettings

settings = CrewSettings()

llm = LLM(
    model=settings.model_name,
    temperature=0.0,
    api_base=settings.model_api_base,
    api_key=settings.model_api_key
)
search_tool = SonarSearchTool()

@CrewBase
class BarcodeLookupCrew():
    """Crew for identifying products by barcode"""
    agents_config = 'src/config/agents.yaml'
    tasks_config = 'src/config/tasks.yaml'

    @agent
    def barcode_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['barcode_researcher'],
            verbose=True,
            llm=llm,
            tools=[search_tool],  
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            llm=llm,
            verbose=True
        )

    @task
    def identify_product_task(self) -> Task:
        return Task(
            config=self.tasks_config['identify_product_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='product_summary.txt'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )