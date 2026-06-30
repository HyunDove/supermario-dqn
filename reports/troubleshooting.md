# 트러블슈팅 기록

> 슈퍼마리오 DQN 프로젝트 (2026-06-25 ~ 2026-07-01)

---

## 01. Colab 환경 의존성 충돌 — numpy 2.0 · gym 0.26 호환성

### 상황

Google Colab T4 GPU 환경에서 학습을 시작하자마자 `AttributeError`, `ValueError` 가 연쇄 발생했습니다.
라이브러리 버전 간 호환성 문제로 학습 자체를 시작할 수 없는 상태였습니다.

### 오류 내용

```
AttributeError: module 'numpy' has no attribute 'bool8'
ValueError: not enough values to unpack (expected 5, got 4)
```

### 원인 분석

| 오류 | 원인 |
|---|---|
| `np.bool8` AttributeError | numpy 2.0에서 `np.bool8` 완전 제거. `nes-py` · `gym` 내부 코드가 이를 사용 중 |
| uint8 오버플로우 | numpy 2.0에서 uint8 자동 형변환 제거 → `self.ram[x] * 256` 연산이 오버플로우 |
| step() ValueError | `gym 0.26`의 `TimeLimit`은 5-tuple `(obs, reward, terminated, truncated, info)` 기대하지만, `gym-super-mario-bros`는 4-tuple 반환 |

세 오류 모두 **numpy 2.0 · gym 0.26의 Breaking Change** 에 서드파티 라이브러리가 아직 대응하지 못한 것이 공통 원인이었습니다.

### 해결

라이브러리 소스 파일을 Colab 설치 직후 **런타임에 일괄 패치**하는 방식으로 해결했습니다.

```python
# numpy bool8 패치
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# uint8 오버플로우 패치 (정규식으로 연산부 int() 캐스팅 삽입)
p1 = re.compile(r'(self\.ram\[[^\]]+\])\s*\*\s*(\d+)')
patched = p1.sub(lambda m: f'int({m.group(1)}) * {m.group(2)}', content)

# gym 0.26 step() 4-tuple/5-tuple 분기 처리
NEW = '''result = self.env.step(action)
if len(result) == 4:
    observation, reward, terminated, info = result
    truncated = False
else:
    observation, reward, terminated, truncated, info = result'''
```

패치 코드는 `colab/train.ipynb` 2번 셀(패키지 설치 직후)에 통합해, 매 세션 시작 시 자동 적용되도록 했습니다.

### 결과

✅ 패치 코드 자동화로 이후 세션에서도 동일 오류 없이 학습 즉시 시작 가능

---

## 02. 영상 녹화 프레임 손상 — C++ 버퍼 참조 문제

### 상황

에피소드 플레이 영상을 MP4로 저장했을 때, 재생하면 화면이 **초록색 노이즈**로 깨지거나 프레임이 손상되는 현상이 발생했습니다.

### 오류 내용

```
저장된 MP4 파일 재생 시 → 초록색 노이즈 또는 첫 프레임만 반복 출력
```

### 원인 분석

`nes-py`의 `render()` 메서드는 내부 C++ 버퍼에 대한 **참조(reference)** 를 반환합니다.
루프가 끝나고 `env.close()` 가 호출되면 해당 메모리가 해제되는데, 이미 `frames` 리스트에 담긴 참조값도 함께 무효화되어 이후 `imageio.mimsave()` 로 저장할 때 손상된 값이 기록되었습니다.

```python
# 문제 코드 — 참조 저장 (env.close() 후 손상)
frame = env.render(mode="rgb_array")
frames.append(frame)
```

### 해결

`render()` 호출 즉시 `np.ascontiguousarray()` 로 **메모리를 복사**해 독립적인 배열로 확보했습니다.

```python
# 수정 코드 — 즉시 복사로 독립 메모리 확보
frame = env.render(mode="rgb_array")
frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
```

### 결과

✅ 모든 에피소드 영상 정상 저장 — `src/utils/recorder.py` 에 적용

---

## 03. Colab nes-py 빌드 실패 — cmake 미설치 · setuptools 충돌

### 상황

Colab에서 `pip install nes-py==8.2.1` 실행 시 C++ 빌드 단계에서 오류가 발생해 설치가 완료되지 않았습니다.

### 오류 내용

```
error: command 'cmake' failed: No such file or directory
Building wheel for nes-py (pyproject.toml) ... error
```

### 원인 분석

`nes-py`는 C++ NES 에뮬레이터 코드를 포함하고 있어 **cmake 빌드**가 필요합니다.
Colab 기본 환경에는 cmake가 설치되어 있지 않았고, 최신 `setuptools`가 빌드 격리(build isolation)를 강제 적용해 cmake를 찾지 못하는 이중 문제가 있었습니다.

### 해결

패키지 설치 셀에 cmake 사전 설치와 setuptools 버전 고정, 빌드 격리 비활성화를 순서대로 적용했습니다.

```bash
# cmake 사전 설치
!apt-get install -y cmake

# setuptools 버전 고정 (빌드 격리 이슈 회피)
!pip install "setuptools>=65.5.1,<82"

# 빌드 격리 비활성화로 설치
!pip install nes-py==8.2.1 --no-build-isolation
```

### 결과

✅ 설치 성공 — `colab/train.ipynb` 1번 셀(패키지 설치)에 순서 고정하여 재현 가능하게 정리

---

## 환경 정보

| 항목 | 내용 |
|---|---|
| 실행 환경 | Google Colab (NVIDIA T4 GPU) |
| Python | 3.12 |
| nes-py | 8.2.1 |
| gym | 0.26.2 |
| gym-super-mario-bros | 7.4.0 |
| PyTorch | 2.x (CUDA 12.x) |
| numpy | 2.x |
