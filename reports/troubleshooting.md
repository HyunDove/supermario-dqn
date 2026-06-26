# 트러블슈팅 기록

---

## [2026-06-25] pip 설치 오류

### 상황
Colab에서 `requirements.txt` 기반 패키지 설치 중 오류 발생

### 오류 내용
```
ERROR: Failed to build 'gym' when getting requirements to build wheel
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```

### 원인
1. `setuptools` 버전 낮음 (57.5.0) → 60.0.0+ 필요
2. `sdv` 패키지가 내부적으로 구버전 `gym` 설치 시도 → Python 3.12에서 빌드 실패
3. Colab 서버는 Python 3.12 사용 (로컬 3.10.6과 다름)

### 해결 방법
`sdv` → `ctgan` 으로 교체 후 셀 순서대로 설치

```python
# 셀 1
!pip install --upgrade setuptools pip

# 셀 2
!pip install torch pandas numpy scikit-learn imbalanced-learn

# 셀 3
!pip install ctgan

# 셀 4
!pip install matplotlib seaborn joblib
```

### import 변경사항
```python
# 기존 (sdv)
from sdv.tabular import CTGAN

# 변경 (ctgan)
from ctgan import CTGAN
```

### 상태
- [ ] 해결 확인 필요 (내일 재시도)

---

## 환경 정보

| 항목 | 내용 |
|---|---|
| 실행 환경 | Google Colab |
| Colab Python | 3.12 |
| 로컬 Python | 3.10.6 |
| GPU | T4 (Colab 무료) |
| 로컬 GPU | Intel Iris Xe (딥러닝 불가) |
