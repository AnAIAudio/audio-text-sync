from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

from prompt import system_prompt

load_dotenv()


def setup_models():
    """모델들을 초기화합니다."""
    models = {
        # "openai": ChatOpenAI(),
        "anthropic": ChatAnthropic(model="claude-3-5-haiku-latest"),
        # "xai": ChatXAI()
    }
    return models


def generate_translate(model, text):
    """단일 모델을 사용하여 번역합니다."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    response = model.invoke(messages)
    return response.content


if __name__ == "__main__":
    text_to_translate = "안녕하세요, 이 텍스트를 영어로 번역해주세요."

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
        SystemMessage(content=system_prompt),
        HumanMessage(content=text_to_translate),
    ]

    # 병렬로 실행
    responses = parallel_chain.invoke(messages)

    # 결과에서 content만 추출
    results = {model_name: response.content for model_name, response in responses.items()}

    for model_name, result in results.items():
        print(f"\n{model_name}: {result}")
