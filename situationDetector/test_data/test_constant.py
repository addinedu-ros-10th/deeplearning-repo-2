
TEST_AI_JSON = {
    "detection" : [
    {
        "class_id": 0,               ## 순찰 이벤트 id
        "class_name": "not_smoking", ## 순찰 이벤트 이름
        "confidence": 1.,            ## detection 신뢰도
        "bbox": {                    ## Box 표기 위치
        "x1": 170.,
        "y1": 481.2,
        "x2": 123.4,
        "y2": 456.7,
        }
    },
    {
        "class_id": 3,               ## 순찰 이벤트 id
        "class_name": "fire",        ## 순찰 이벤트 이름
        "confidence": 0.5,           ## detection 신뢰도
        "bbox": {                    ## Box 표기 위치
        "x1": 170.,
        "y1": 481.2,
        "x2": 123.4,
        "y2": 456.7,
        }
    }
    ],
    "detection_count": 2,
    "timestamp": "2025-09-18 14:00:00",
    "patrol_number": 0,
}

