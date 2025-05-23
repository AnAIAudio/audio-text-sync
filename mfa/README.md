설치방법 (Anaconda or Miniconda)
```
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

아래의 에러가 날 시 joblib==1.1.0 설치 필요
```
TypeError: init() got an unexpected keyword argument 'cachedir'
```

mfa version 명령어를 통해 제대로 설치됬는지 확인

아래의 명령어를 통해 영어 음향모델, 발음 사전 설치
```
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

강제 정렬을 위해 사용하는 명령어
```
mfa align [데이터 디렉토리] [사전 파일] [음향 모델(.zip)] [출력 디렉토리] [옵션들]
```

아래의 에러가 날 시 beam, retry_beam 을 설정 혹은 늘려서 다시 실행
```
montreal_forced_aligner.exceptions.NoAlignmentsError: NoAlignmentsError
```

자주 사용하는 파라미터 ([참고](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html#pretrained-alignment))

- --num_jobs
병렬 처리할 작업 수 (기본 3)
- --output_format
출력 형식 지정 (기본은 long_textgrid)
  - long_textgrid (기본값): 전체 문장을 하나의 라인으로
  - short_textgrid: 단어, 음소별로 분리된 트랙
  - json: 시간 정보 포함 JSON
  - csv: 라벨과 시간 정보가 CSV로 출력됨
- --clean
기존 출력 디렉토리 내용을 제거하고 새로 실행 (기본 False)
- --overwrite
기존 작업 결과가 있더라도 덮어쓰기
- --debug
로그를 자세히 출력
- --beam
탐색 너비 조절 (기본: 10, 더 높이면 정밀하지만 느림)
- --retry_beam
1차 실패 시 더 넓게 재시도할 때 사용 (기본: 40)
- --boost_silence
  - 무음(silence)의 점수를 얼마나 높여줄지
  - 값이 높을수록 무음으로 처리되는 구간이 많아짐
- --duration_threshold
정렬 결과 중 너무 짧은 구간은 무시할 수 있도록 설정
  - ex) --duration_threshold 0.01 → 10ms 이하 무시
- --use_mp
    - 멀티 프로세싱을 사용하는 설정
- --use_threading
    - 스레딩을 사용하는 설정
- --config_path
YAML 형식의 설정 파일을 사용해 세부 파라미터를 지정
    ```
    beam: 100
    retry_beam: 400
    num_jobs: 12
    ```