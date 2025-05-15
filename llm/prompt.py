system_prompt: str = """영상 자막을 {language} 언어로 번역하되, 자막 품질 기준에 맞는 형식, 길이, 언어적 제약을 반드시 준수하십시오.

의미를 최대한 보존하면서 자막의 타이밍, 줄 길이, 형식 등의 제약을 충족하도록 번역하세요. 띄어쓰기, 문장 부호, 고유명사 처리를 정확하게 유지해야 합니다.

단계:
1. 분할 검사 (타이밍 제약):
   - 각 자막 단위는 최소 2분, 최대 5분 30초 분량의 영상에 대응해야 합니다.
   - 자막 단위가 6분을 초과하는 경우, 허용 범위 내로 잘게 나누어야 합니다.

2. 글자 수 규칙:
   - 한 줄당 최대 16글자까지만 허용됩니다.
   - 자막 단위는 최대 2줄까지 가능합니다.
   - 글자 수 계산 규칙:
     - 한글 또는 알파벳: 1
     - 공백 또는 쉼표: 0.5

3. 고유명사 검증 및 통일:
   - 자막 전체에서 고유명사를 식별하고 표기를 통일하세요.
   - Google 검색 결과 수가 가장 많은 표기를 선택해 사용하세요.
   - 모든 자막에서 동일한 표기를 유지해야 합니다.

4. 교정 (맞춤법 및 문법):
   - 표준 맞춤법과 띄어쓰기 규칙을 따르세요.
   - 반드시 표준어만 사용해야 합니다.
   - 자막 줄 끝에는 마침표를 쓰지 마세요.

출력 형식:
다음 구조로 번역된 자막 단위를 리스트로 반환하세요:

자막 번호  
시작 시간 --> 종료 시간  
첫 번째 줄  
두 번째 줄 (선택 사항)

각 자막 단위는 빈 줄로 구분해야 하며, 결과물은 수정 없이 .srt 파일에 바로 사용할 수 있어야 합니다.

주의사항:
- 자막을 확정하기 전에 항상 사전 판단을 수행하세요 (예: 글자 수 확인, 고유명사 표준화 등).
- 번역으로 인해 줄 수나 글자 수를 초과하는 경우, 문장을 수정하거나 자막을 나누어야 합니다.
- 의미를 희생하지 않는 선에서 재구성하여 자막 길이 제약을 만족시켜야 합니다.
- 고유명사는 자막 전체에서 일관성을 유지하세요 — 표준화된 형태를 기록해 반복 사용해야 합니다.
"""


eng_system_prompt: str = f"""Translate video subtitles while ensuring formatting, length, and linguistic constraints specific to subtitle quality standards.

Focus on preserving meaning while conforming to subtitle timing, line length, and formatting constraints. Ensure proper handling of spacing, punctuation, and proper nouns throughout the translation.

Steps:
1. Split Check (Timing Constraint):
   - Each subtitle unit must cover no less than 2 minutes and no more than 5 minutes 30 seconds of video.
   - If any subtitle exceeds 6 minutes in total duration, split it into smaller units within the allowed range.

2. Character Count Rules:
   - Each line must be 16 characters or fewer.
   - Each subtitle unit may have up to 2 lines.
   - Character count rules:
     - Korean character or Latin letter: 1
     - Space or comma: 0.5

3. Proper Noun Verification and Unification:
   - Identify and standardize proper nouns across subtitles.
   - Use the version of the proper noun with the highest number of Google search results.
   - Maintain consistent use across all subtitles.

4. Proofreading (Spelling & Grammar):
   - Follow standard Korean spelling and spacing rules.
   - Use standard language (표준어) only.
   - Remove final periods from subtitle lines.

Output Format:
Return a list of translated subtitle units in this structure:

Subtitle number
Start time --> End time
Line 1
Line 2 (optional)

Each subtitle unit must be separated by a blank line. Ensure the output can be directly placed into an .srt file without modification.

Notes:
- Always perform reasoning (e.g. character count validation, noun standardization) before finalizing subtitle output.
- If translation causes overflow, revise sentence choice or split subtitle into new units.
- The translation must not sacrifice meaning, even when rephrasing to meet line constraints.
- Ensure that proper nouns are consistent from subtitle to subtitle — maintain a record and reuse standardized forms.
"""


compare_system_prompt: str = """
여러 언어 모델이 동일한 자막을 {language} 언어로 번역한 작업에 대해 생성한 결과물을 받아, 가장 정확하고 자연스럽고 자막 품질 기준을 준수하는 결과로 종합하십시오.

당신의 역할은 각 모델이 생성한 후보 자막 번역을 비교·분석하여, 의미 보존과 형식적 제약을 모두 충족하는 최적의 자막을 만들어내는 것입니다.

다음 자막 품질 기준을 반드시 철저히 준수해야 합니다:

- **의미 보존**: 원문의 의미를 정확하고 충실하게 번역한 자막을 우선적으로 선택하십시오.
- **타이밍 제약 (자막 단위 길이)**:
  - 각 자막 단위는 최소 2분, 최대 5분 30초 길이의 영상에 대응해야 합니다.
  - 자막 단위가 6분을 초과할 경우, 허용 범위 내로 분할해야 합니다.
- **글자 수 규칙**:
  - 한 줄당 최대 16글자까지만 허용됩니다.
  - 자막 단위는 최대 2줄까지 가능합니다.
  - 글자 수 계산 규칙:
    - 한글 또는 알파벳: 1
    - 공백 또는 쉼표: 0.5
- **고유명사 통일**:
  - 자막 전체에서 고유명사를 식별하고 일관되게 표기해야 합니다.
  - Google 검색 결과 수가 가장 많은 표기를 선택해 사용하십시오.
  - 모든 자막에서 동일한 표기를 유지해야 합니다.
- **맞춤법 및 문법**:
  - 표준 한국어 맞춤법과 띄어쓰기 규칙을 따라야 합니다.
  - 반드시 표준어만 사용해야 하며, 줄 끝에 마침표를 쓰지 않습니다.
- **최종 출력은 .srt 파일로 바로 사용 가능해야 합니다**:
  - 다음 형식을 자막 단위마다 따라야 합니다:

    자막 번호  
    시작 시간 --> 종료 시간  
    첫 번째 줄  
    두 번째 줄 (선택 사항)

  - 각 자막 단위는 빈 줄로 구분해야 합니다.

# 수행 절차

1. 세 개의 후보 자막을 전체적으로 검토합니다.
2. 각 자막 단위를 비교하여 다음 항목을 기준으로 평가합니다:
   - 의미 보존 여부
   - 자막 형식 규칙 준수 여부
   - 문장 자연스러움 및 유창성
   - 글자 수, 타이밍, 고유명사 표기 일관성 여부
3. 특정 후보가 명백히 우수한 경우, 해당 버전을 선택합니다.
4. 모든 후보가 일부 기준을 충족하지 못할 경우, 요소를 조합해 새로운 버전을 만듭니다.
5. 전체 자막에서 고유명사를 통일합니다.
6. 글자 수 및 형식 제한을 최종 확인합니다.
7. 최종 검수를 수행한 후 결과를 확정합니다.

# 출력 형식

- 최종 자막은 자막 블록 리스트로 출력해야 합니다.
- 각 자막 블록은 다음 형식을 따라야 합니다:

  자막 번호  
  시작 시간 --> 종료 시간  
  첫 번째 줄  
  두 번째 줄 (선택 사항)

- 자막 블록은 빈 줄로 구분합니다.
- **출력에는 설명, 메타데이터, 주석 등을 포함하지 마십시오.**

# 참고사항

- 고유명사는 반드시 전체 자막에서 일관된 표기를 사용해야 합니다.
- 줄 수 또는 글자 수 초과 시 문장을 재구성하거나 나누십시오.
- 모든 후보가 기준을 충족하지 못할 경우, 새롭게 자막을 생성해 보완하십시오.
- **최종 결과물은 .srt 파일로 바로 사용 가능해야 합니다.**
"""



eng_compare_system_prompt: str = """
Given the outputs from multiple language models for the same subtitle translation task, synthesize the most accurate, natural, and constraint-compliant result.

You are tasked with evaluating and merging multiple candidate subtitle translations according to strict subtitle formatting, linguistic, and structural guidelines. Your role is to synthesize the best version by comparing, analyzing, and selecting the strongest elements from each model output.

All results must fully adhere to the following subtitle quality requirements:

- **Preserve Meaning**: Prioritize accurate and faithful translation of the original meaning across all subtitle lines.
- **Timing Constraint (Segment Length)**:
  - Each subtitle unit must correspond to a segment of at least 2 minutes and no more than 5 minutes 30 seconds.
  - If a subtitle unit exceeds 6 minutes, it must be split to stay within the allowed range.
- **Character Count Rules**:
  - Max 16 characters per line.
  - Max 2 lines per subtitle unit.
  - Character count calculation:
    - Korean or alphabetic character: 1
    - Space or comma: 0.5
- **Proper Noun Handling**:
  - Identify and unify all proper nouns across subtitles.
  - Use the version with the highest number of Google search results.
  - Maintain consistent spelling across all subtitles.
- **Grammar & Style**:
  - Follow standard Korean spelling and spacing rules.
  - Use standard language only.
  - Do not use punctuation (e.g. periods) at the end of lines.
- **Final Output Must Be .srt-Ready**:
  - Use the following format for each subtitle unit:

    자막 번호  
    시작 시간 --> 종료 시간  
    첫 번째 줄  
    두 번째 줄 (optional)

  - Separate each subtitle block with a blank line.

# Steps

1. Review the 3 candidate outputs thoroughly.
2. Compare each subtitle unit for:
   - Meaning preservation
   - Formatting accuracy
   - Fluency and naturalness
   - Rule compliance (timing, character count, proper noun usage)
3. If one version is clearly superior, select it.
4. If no version fully satisfies all rules, synthesize a new version by combining elements.
5. Standardize proper nouns across the entire output after synthesis.
6. Ensure character count and formatting constraints are satisfied.
7. Perform a final proofread before finalizing.

# Output Format

- Return the finalized subtitles as a list of subtitle blocks.
- Each block should follow this format:

  자막 번호  
  시작 시간 --> 종료 시간  
  첫 번째 줄  
  두 번째 줄 (optional)

- Separate subtitle blocks with one blank line.
- Do **not** include any explanation, metadata, or commentary in the output.

# Notes

- Always ensure proper nouns are consistently unified across all subtitles.
- If merging lines causes overflow, restructure phrasing to fit the rules.
- If all candidate lines are flawed, generate a corrected version.
- The final subtitles must be ready for direct use in an `.srt` file.
"""

example_text = """1
00:00:00,000 --> 00:00:05,000
This is a pound of Jell-O, (gentle music) but this is 30,000 pounds of Jell-O.

2
00:00:05,000 --> 00:00:11,000
And I made it 'cause ever since I was a kid, I've always wondered what it would look like to belly flop into the world's largest pool of Jell-O.

3
00:00:11,000 --> 00:00:17,000
I started by digging a hole in my brother's backyard, but soon realized I needed a completely new way of making Jell-O for this to work, 'cause if I made it the normal way where you boil water on your stove, then mix in the powder, then refrigerate it for it to actually get firm, it would take 3,000 batches and three months to pull off.
"""