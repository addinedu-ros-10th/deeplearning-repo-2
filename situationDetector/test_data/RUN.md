## 실행 방법

1. 가상환경 활성화

```
source .venv/bin/activate
```

2. ped_main 실행 (터미널1)

```
python -m deviceManager.ped_main
```


3. sd_main (situationDetector_main) 실행 (터미널2)

```
python -m situationDetector.sd_main
```

4. 더미 dataService 실행 (단순 출력기능, 터미널3)

```
python -m situationDetector.test_tcp_ds_receive
```