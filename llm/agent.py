from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

from prompt import system_prompt, compare_system_prompt, example_text

load_dotenv()


def setup_models():
    """모델들을 초기화합니다."""
    models = {
        # "openai": ChatOpenAI(model="gpt-4.1-nano"),
        "anthropic": ChatAnthropic(model="claude-3-5-haiku-latest"),
        # "xai": ChatXAI(model="grok-3-mini-latest")
    }
    return models


def run_agent(text_to_translate: str = example_text, language: str = "한국어") -> dict[str, str]:
    print("\n=== LangChain 병렬 번역 ===")

    models = setup_models()

    # 각 모델에 대한 chain 생성
    chains = {}
    for model_name, model in models.items():
        chains[model_name] = model

    # 병렬 실행을 위한 RunnableParallel 생성
    parallel_chain = RunnableParallel(chains)

    # 모든 모델에 동일한 메시지 전달
    messages = [
        SystemMessage(content=system_prompt, language=language),
        HumanMessage(content=text_to_translate),
    ]

    # 병렬로 실행
    responses = parallel_chain.invoke(messages)

    # 결과에서 content만 추출
    results = {model_name: response.content for model_name, response in responses.items()}

    for model_name, result in results.items():
        print(f"\n{model_name}: \n{result}")

    # 3개의 모델에서 반환된 결과값에 대해 최종 결과물을 종합하기 위한 모델 생성
    compare_model = ChatAnthropic(model="claude-3-5-haiku-latest")

    # compare_system_prompt 으로 변경
    messages = [
        SystemMessage(content=compare_system_prompt, language=language),
        HumanMessage(content=f"{results}"),
    ]

    responses = compare_model.invoke(messages)
    print(f"\ncompare_version: \n{responses.content}")

    results["compare_version"] = responses.content
    return results