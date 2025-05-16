import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

from audio.cut_wave import write_srt
from llm.prompt import system_prompt, compare_system_prompt, example_text
from text.text_util import Segment

load_dotenv()


class AgentModel:
    def __init__(
        self,
        custom_system_prompt: str = system_prompt,
        custom_compare_system_prompt: str = compare_system_prompt,
    ):
        self.system_prompt = custom_system_prompt
        self.compare_system_prompt = custom_compare_system_prompt

    def setup_models(self):
        """모델들을 초기화합니다."""
        models = {
            # "openai": ChatOpenAI(model="gpt-4.1-nano"),
            "anthropic": ChatAnthropic(model="claude-3-5-haiku-latest"),
            # "anthropic_v2": ChatAnthropic(model="claude-3-5-haiku-latest"),
            # "anthropic_v3": ChatAnthropic(model="claude-3-5-haiku-latest"),
            # "xai": ChatXAI(model="grok-3-mini-latest")
        }
        return models

    def segments_to_list(self, segments: list[Segment]) -> list[str]:
        from datetime import timedelta

        list_segments = []
        for id, segment in enumerate(segments, start=1):
            start_time = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
            end_time = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
            text = segment["text"]

            if not text:
                continue

            text = text[1:] if text[0] == " " else text
            segment = f"{id}\n{start_time} --> {end_time}\n{text}\n\n"
            list_segments.append(segment)

        print(list_segments)
        return list_segments

    def segments_generator(self, segments: list[str], n: int = 1):
        gen_segments = []
        for idx, segment in enumerate(segments, start=1):
            gen_segments.append(segment)

            if idx % n == 0:
                yield gen_segments
                gen_segments = []

        if gen_segments:
            yield gen_segments

    def run(
        self,
        srt_directory_path: str,
        formatted: str,
        segments: list[Segment],
        language: str = "한국어",
        seperate_number: int = 50,
    ) -> dict[str, str]:
        print("\n=== LangChain 병렬 번역 ===")

        models = self.setup_models()

        # 각 모델에 대한 chain, 최종 결과값을 담는 딕셔너리 생성
        chains = {}
        results = {}
        for model_name, model in models.items():
            chains[model_name] = model
            results[model_name] = ""

        # 병렬 실행을 위한 RunnableParallel 생성
        parallel_chain = RunnableParallel(chains)

        # segments 을 n 개씩 나누기 위해 수행
        list_segments = self.segments_to_list(segments)
        for segment in self.segments_generator(list_segments, n=seperate_number):
            text_to_translate = "".join(segment)
            print(f"\n{text_to_translate}")

            # 모든 모델에 동일한 메시지 전달
            messages = [
                SystemMessage(content=self.system_prompt, language=language),
                HumanMessage(content=text_to_translate),
            ]

            # 병렬로 실행
            responses = parallel_chain.invoke(messages)

            # 결과에서 content만 추출
            for model_name, response in responses.items():
                results[model_name] += response.content
                print(f"\n{model_name}: {response.content}")

        # 결과값 srt 파일로 저장 및 출력
        for model_name, result in results.items():
            srt_file_path = os.path.join(
                srt_directory_path, f"voix_result_srt_{model_name}_{formatted}.srt"
            )
            write_srt(srt_file_path=srt_file_path, text=result)

            print(f"\n{model_name}: \n{result}")

        # 3개의 모델에서 반환된 결과값에 대해 최종 결과물을 종합하기 위한 모델 생성
        compare_model = ChatAnthropic(model="claude-3-5-haiku-latest")

        # compare_system_prompt 으로 변경
        messages = [
            SystemMessage(content=self.compare_system_prompt, language=language),
            HumanMessage(content=f"{results}"),
        ]

        responses = compare_model.invoke(messages)
        print(f"\ncompare_version: \n{responses.content}")

        results["compare_version"] = responses.content
        srt_file_path = os.path.join(
            srt_directory_path, f"voix_result_srt_compare_{formatted}.srt"
        )
        write_srt(srt_file_path=srt_file_path, text=results["compare_version"])

        return results
