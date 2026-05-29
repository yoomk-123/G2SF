# G²SF 재현 실험 프롬프트 로그 (PROMPTS.md)

> **논문**: G²SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection (ICCV 2025)
> **사용 도구**: Claude Code
> **GitHub**: https://github.com/yoomk-123/G2SF

---

## 1단계 · 환경 구축

**[Prompt]**
로컬 JupyterHub GPU 환경에서 G²SF 실행 시 CUDA toolkit이 설치되지 않는 상황에서 conda 환경 충돌이 발생하고 있습니다. 어떻게 해결할 수 있을까요?

**[핵심 해결]**
- Step 3~4에서 충돌 발생 확인
- conda 환경 생성 시 `cuda-cccl=12.1` 버전 고정 필수
- 미고정 시 conda가 최신 버전(12.9)을 설치하면서 nvcc 12.1과 버전 불일치 발생
- `fatal error: nv/target: No such file or directory` 오류 발생

---

**[Prompt]**
시스템에 CUDA toolkit이 없는 환경에서 pointnet2 CUDA 모듈 빌드를 어떻게 성공시킬 수 있을까요?

**[핵심 해결]**
conda 환경 내부 경로를 환경변수로 명시적 설정

```python
CONDA_ENV_PATH = '/opt/conda/envs/G2SF'
env['CUDA_HOME']            = CONDA_ENV_PATH
env['TORCH_CUDA_ARCH_LIST'] = '8.6'   # RTX A6000
env['FORCE_CUDA']           = '1'

INCLUDE = (f"{CONDA_ENV_PATH}/include:"
           f"{CONDA_ENV_PATH}/targets/x86_64-linux/include")
env['CPATH'] = env['C_INCLUDE_PATH'] = env['CPLUS_INCLUDE_PATH'] = INCLUDE
```

---

## 2단계 · 소스코드 패치

**[Prompt]**
재현 실행 중 발생한 버그들을 일괄 패치하고 싶습니다. 발견된 문제들을 정리하고 자동으로 수정하는 코드를 작성해주세요.

**[핵심 해결]**
총 5가지 버그 일괄 패치

| 패치 | 대상 파일 | 원인 | 해결 |
|------|----------|------|------|
| 패치 1 | `imgaug.py` | numpy 2.0에서 `np.sctypes` API 제거 | 명시적 타입 집합으로 교체 |
| 패치 2 | `Engine/train.py` | 미정의 클래스 `UNSFusionModel` import | import 구문 제거 |
| 패치 3 | `Model/models.py` | `./Checkpoint/` vs `./Checkpoints/` 불일치 | 경로 수정 |
| 패치 4 | `Dataset/__init__.py` | 원본은 10개 클래스 전체 실행 | `cable_gland` 단독 실행 설정 |
| 패치 5 | `Engine/train.py` 외 | `./Result/` vs `./Results/` 불일치 | 전체 소스 일괄 수정 |

```python
# 패치 4 예시: cable_gland 단독 실행
new_func = 'def mvtec3d_classes():\n    return ["cable_gland"]'
p = re.sub(r'def mvtec3d_classes\(\):\s+return \[.*?\]',
           new_func, c, flags=re.DOTALL)

# 패치 5 예시: Result → Results 전체 소스 일괄
for f in CODE_DIR.rglob('*.py'):
    c = f.read_text()
    if './Result/' in c:
        f.write_text(c.replace('./Result/', './Results/'))
```

---

## 3단계 · 가중치 및 데이터셋 준비

**[Prompt]**
DINOv2, PointMAE, G²SF 사전학습 가중치를 자동으로 다운로드하고 압축 해제하는 코드를 작성해주세요.

**[핵심 해결]**
- DINOv2, PointMAE: `gdown` 라이브러리로 Google Drive 자동 다운로드
- G²SF 가중치: RAR/ZIP 형식 자동 감지 후 압축 해제
- 파일 헤더(`header[:4] == b'Rar!'`)로 형식 자동 판별

```python
import gdown

def download(fid, out, label):
    p = Path(out)
    if p.exists() and p.stat().st_size > 1e5:
        print(f'이미 존재: {label} ({p.stat().st_size/1e6:.0f} MB)')
        return True
    gdown.download(id=fid, output=str(p), quiet=False)

download('14vQqN4Do1Vnx2TZVJ16LAW81FRq3o-UQ',
         CKPT_DIR / 'dino_vitbase8_pretrain.pth', 'DINOv2')
download('14d04kH3bX2BbDEIJPI4MkRtolfQw_fds',
         CKPT_DIR / 'pointmae_pretrain.pth', 'PointMAE')
```

---

**[Prompt]**
MVTec3D-AD 데이터셋 tar.xz 파일을 자동으로 탐색하고 압축 해제하는 코드를 작성해주세요. 디스크 용량 문제로 cable_gland 클래스만 사용합니다.

**[핵심 해결]**
- `rglob`으로 하위 폴더까지 tar 파일 자동 탐색
- `tarfile` 라이브러리로 `.tar.xz` 압축 해제

```python
tar_candidates = (list(DATASET_DIR.rglob('*.tar')) +
                  list(DATASET_DIR.rglob('*.tar.gz')) +
                  list(DATASET_DIR.rglob('*.tar.xz')))
cable_tars = [t for t in tar_candidates if 'cable' in t.name.lower()]

with tarfile.open(tar_path) as tf:
    tf.extractall(str(MVTEC_DIR), filter='data')
```

---

## 4단계 · 추론 실행 및 결과 분석

**[Prompt]**
추론 실행 중 출력되는 로그에서 모달별 수치 결과를 자동으로 수집하고 pandas DataFrame으로 정리해주세요.

**[핵심 해결]**
추론 로그 실시간 파싱 → pandas DataFrame 정리

```python
# 추론 중 modal_lines 자동 수집
for line in proc.stdout:
    print(line, end='')
    if 'sample-level auc' in line:
        modal_lines.append(line)

# 결과 파싱
m = re.match(
    r'Modal (\S+):\s+sample-level auc ([\d.]+), pixel-level auc ([\d.]+), '
    r'aupro ([\d.]+), aupro_0p01 ([\d.]+), aupro_0p1 ([\d.]+), aupro_0p05 ([\d.]+)', line)

df = pd.DataFrame(rows).set_index('Modal')
df = (df * 100).round(1)
```

---

**[Prompt]**
결함 타입별 Anomaly Score Map을 시각화하고 저장하는 코드를 작성해주세요.

**[핵심 해결]**
`matplotlib`으로 결함 타입별 Anomaly Map 시각화

```python
result_base = RESULT_DIR / 'mvtec' / 'Complete' / 'cable_gland'
defect_types = [d.name for d in sorted(result_base.iterdir())
                if d.is_dir() and d.name != 'fusion']

fig, axes = plt.subplots(len(defect_types), n_cols,
                         figsize=(18, 5 * len(defect_types)))
for row, dtype in enumerate(defect_types):
    imgs = sorted((result_base / dtype).glob('*.jpg'))[:n_cols]
    for col, img in enumerate(imgs):
        axes[row][col].imshow(plt.imread(str(img)))
        axes[row][col].set_title(f'{dtype}/{img.name}', fontsize=9)
```

---

## 5단계 · 노트북 리팩토링

**[Prompt]**
재현 과정에서 디버깅을 위해 추가된 셀들이 많아 노트북이 지저분합니다. 불필요한 셀을 제거하고 처음 실행하는 환경에서도 전체 파이프라인이 자동으로 구성되도록 리팩토링해주세요.

**[핵심 해결]**
- 디버깅용 셀 전부 제거
- 각 단계 자동 스킵 로직 추가 (이미 완료된 경우 재실행 방지)
- Step 8-A/B 구분 명확화 (`--load_feature False/True`)
- Step 9 결과 수집 및 시각화 분리
- `G2SF_Final.ipynb` 생성

---

## 실험 결과 요약

| Metric | Original Paper | Ours (Reproduction) | Difference |
|--------|---------------|---------------------|------------|
| I-AUROC | 97.1 | 92.3 | -4.8 |
| P-AUROC | 99.7 | 99.1 | -0.6 |
| AUPRO@30% | 97.7 | 96.8 | -0.9 |
| AUPRO@1% | 47.1 | 41.9 | -5.2 |

> 단일 클래스 실행으로 인한 anomaly synthesis 다양성 제한 및 GPU 아키텍처 차이로 일부 수치 차이가 있으나,
> P-AUROC 99.1%로 논문(99.7%) 대비 우수한 성능 재현 확인
