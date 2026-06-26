# 트러블슈팅 기록

> 슈퍼마리오 DQN 프로젝트 (2026-06-25 ~)

---

## [2026-06-25] nes-py 빌드 오류

### 상황
Colab에서 `nes-py==8.2.1` 설치 중 빌드 실패

### 오류 내용
```
error: command 'cmake' failed: No such file or directory
Building wheel for nes-py (pyproject.toml) ... error
```

### 원인
- Colab 기본 환경에 `cmake` 미설치
- `setuptools` 버전이 너무 높으면 `--no-build-isolation` 없이 빌드 실패

### 해결
```python
!apt-get install -y cmake
!pip install "setuptools>=65.5.1,<82"
!pip install nes-py==8.2.1 --no-build-isolation
```

### 상태
✅ 해결 완료

---

## [2026-06-25] numpy 2.0 호환성 — np.bool8 제거

### 상황
학습 실행 중 `AttributeError` 발생

### 오류 내용
```
AttributeError: module 'numpy' has no attribute 'bool8'
```

### 원인
numpy 2.0에서 `np.bool8` 완전 제거. `nes-py`, `gym` 내부 코드가 이를 사용

### 해결
```python
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
```

패키지 소스 파일도 직접 패치:
```python
for path in glob.glob('/usr/local/lib/python*/dist-packages/gym/utils/passive_env_checker.py'):
    with open(path) as f:
        content = f.read()
    patched = content.replace('np.bool8', 'np.bool_')
    with open(path, 'w') as f:
        f.write(patched)
```

### 상태
✅ 해결 완료 — 패치 코드 `colab/train.ipynb` 2번 셀에 포함

---

## [2026-06-25] numpy 2.0 호환성 — uint8 오버플로우

### 상황
마리오 환경 실행 중 게임 상태값 계산 오류

### 원인
numpy 2.0에서 uint8 연산 시 자동 형 변환이 제거됨
`self.ram[x] * 256` 같은 코드가 오버플로우 발생

### 해결
`nes_py`, `gym_super_mario_bros` 소스 파일 일괄 패치:
```python
p1 = re.compile(r'(self\.ram\[[^\]]+\]|self\.\w+_size)\s*\*\s*((?:0x[0-9a-fA-F]+|\d+)(?:\*\*\d+)?)')
p2 = re.compile(r'(?<!int\()self\.ram\[([^\]:]+)\](?!\s*[=\[])')

patched = p1.sub(lambda m: f'int({m.group(1)}) * {m.group(2)}', content)
patched = p2.sub(r'int(self.ram[\1])', patched)
```

### 상태
✅ 해결 완료 — 패치 코드 `colab/train.ipynb` 2번 셀에 포함

---

## [2026-06-25] gym 0.26 호환성 — step() 반환값 불일치

### 상황
`gym.wrappers.TimeLimit`에서 `ValueError` 발생

### 오류 내용
```
ValueError: not enough values to unpack (expected 5, got 4)
```

### 원인
`gym-super-mario-bros`의 `JoypadSpace.step()`은 4-tuple 반환 `(obs, reward, done, info)`
`gym 0.26`의 `TimeLimit`은 5-tuple `(obs, reward, terminated, truncated, info)` 기대

### 해결
`time_limit.py` 직접 패치:
```python
OLD = 'observation, reward, terminated, truncated, info = self.env.step(action)'
NEW = '''result = self.env.step(action)
        if len(result) == 4:
            observation, reward, terminated, info = result
            truncated = False
        else:
            observation, reward, terminated, truncated, info = result'''
```

### 상태
✅ 해결 완료 — 패치 코드 `colab/train.ipynb` 2번 셀에 포함

---

## [2026-06-25] 영상 녹화 화면 깨짐

### 상황
`imageio.mimsave()`로 저장한 MP4를 열면 화면이 깨지거나 초록색 노이즈 출력

### 원인
`nes-py`의 `render()`가 C++ 내부 버퍼 참조를 반환 → `env.close()` 후 메모리 해제되어 프레임 손상

### 해결
```python
frame = env.render(mode="rgb_array")
frames.append(np.ascontiguousarray(frame, dtype=np.uint8))  # 즉시 복사
```

### 상태
✅ 해결 완료 — `src/utils/recorder.py`에 적용

---

## [2026-06-25] 영상 재생 속도 / 길이 문제

### 상황
녹화된 MP4가 느리게 재생되거나 1분 30초 이상으로 너무 김

### 원인
- NES는 ~60fps, SkipFrame(4) 적용 시 실제 스텝 = 15fps
- `target_steps` 조건을 에피소드 단위로 체크해 에피소드가 끝나야 중단 → 과도하게 길어짐

### 해결
```python
_RECORD_FPS = 15  # NES 60fps ÷ SkipFrame 4

# 루프 내부에서 매 스텝마다 target_steps 체크
while not done and total_steps < target_steps:
    frame = env.render(mode="rgb_array")
    frames.append(...)
    ...

# RECORD_STEPS=450 → 30초, RECORD_SPEED=2.0 → 2배속 저장
imageio.mimsave(save_path, frames, fps=_RECORD_FPS * playback_speed)
```

### 상태
✅ 해결 완료 — `src/utils/recorder.py` 및 노트북 설정값 반영

---

## [2026-06-26] matplotlib 한글 폰트 깨짐

### 상황
Colab에서 저장한 그래프 PNG에 한글이 □□□로 출력

### 원인
Colab 기본 matplotlib는 한글 폰트 미포함. `rcParams` 전역 설정이 셀 재실행 시 초기화됨

### 해결
그래프 저장 함수 내부에서 매번 폰트 재적용:
```python
def _setup_korean_font():
    _NANUM_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(_NANUM_PATH):
        fm.fontManager.addfont(_NANUM_PATH)
        prop = fm.FontProperties(fname=_NANUM_PATH)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False
```

패키지 설치 셀에 `fonts-nanum` 추가:
```bash
!apt-get install -y fonts-nanum
```

### 상태
✅ 해결 완료 — 학습 셀 및 패키지 설치 셀에 적용

---

## 환경 정보

| 항목 | 내용 |
|---|---|
| 실행 환경 | Google Colab |
| Colab Python | 3.12 |
| GPU | NVIDIA T4 (Colab 무료) |
| nes-py | 8.2.1 |
| gym | 0.26.2 |
| gym-super-mario-bros | 7.4.0 |
| PyTorch | 2.x (CUDA 12.x) |
| numpy | 2.x |
