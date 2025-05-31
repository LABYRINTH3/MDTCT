# MDTCT (Masked Diffusion Transformer with Curriculum Training)
**Diffusion 방식을 활용해 만든 언어 모델입니다.** 

전통적인 모델은 autoregressive 방식을 이용해 느리고 오류가 누적되기 쉬운데 MDTCT는 문장 전체를 마스크 토큰으로 바꾼 뒤 가려진 모든 단어를 병렬로 복원하도록 트랜스포머를 훈련시켰습니다.

### 프로젝트의 목적

- **autoregressive text generation의 한계**
    - autoregressive text generation 의 가진 긴 시퀀스에서 느리고 오류가 누적되기 쉬운 단점 해결
- **diffusion model의 확장**
    - 이미지나 오디오 도메인에서는 반복적 노이즈 제거를 수행하는데 텍스트에도 노이즈 과정을 도입할 수 있을까에 대한 의문을 가지게 되었습니다.

### 구현

- **모델**
    - 전체 시퀀스에 마스크 토큰을 포함해 입력받아, 마스킹된 위치를 동시에 예측하는 Transformer 인코더(6개 레이어, 8개 헤드, d_model=512).
- **커리큘럼 학습**
    - 두 개의 데이터셋을 마스크 비율 p=0.2~0.8단계로 순차 훈련.
    1. 샤드된 Arrow 파일 로드, 90% 훈련, 10% 검증 분할
    2. 에포크 2회, 배치 사이즈 256, 학습률 1e-4, 선형 워밍업 스케줄러 사용
    3. 각 단계별 체크포인트 저장 후 다음 단계 초기화
- **추론(Inference)**: `sample_from_model(prompt, R, T=40)`
    1. 프롬프트 뒤에 R개의 마스크 추가
    2. 40단계에 걸쳐 마스크 비율 1→0 선형 감소
    3. 각 단계에서 토큰 일부를 재마스킹 후 병렬 예측, top-1 로짓으로 채워넣기
    4. 중간 출력을 기록해 텍스트 “결정화(crystallization)” 과정을 시각화
