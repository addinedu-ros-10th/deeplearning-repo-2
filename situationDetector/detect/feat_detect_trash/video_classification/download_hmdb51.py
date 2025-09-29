# download_hmdb51.py
import fiftyone as fo
import fiftyone.zoo as foz
import os # os 모듈을 사용하기 위해 추가

print("HMDB51 데이터셋 다운로드를 시작합니다. 시간이 다소 걸릴 수 있습니다...")

dataset = foz.load_zoo_dataset(
    "hmdb51",
    split="test"
)

dataset.persistent = True

print("\n" + "="*50)
print("HMDB51 데이터셋 다운로드 및 설정 완료!")

# 첫 번째 샘플의 파일 경로에서 데이터셋의 루트 디렉토리를 알아냅니다.
if dataset.first():
    first_sample_path = dataset.first().filepath
    # 예: /home/momo/fiftyone/hmdb51/test/data/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    # 위 경로에서 상위 폴더로 이동하여 기본 경로를 찾습니다.
    dataset_root_dir = os.path.abspath(os.path.join(first_sample_path, "..", "..", ".."))
    print(f"데이터셋 위치: {dataset_root_dir}")
else:
    print("데이터셋이 로드되었지만 샘플이 없습니다. fiftyone 기본 폴더를 확인하세요: ~/fiftyone/hmdb51")

print("이제 이 폴더 안에서 클래스별 비디오 파일을 사용할 수 있습니다.")
print("="*50)