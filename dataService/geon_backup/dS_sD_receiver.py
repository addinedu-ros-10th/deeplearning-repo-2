import socket
import json
import mysql.connector
from mysql.connector import Error

# --- 1. DB 연결 정보 설정 ---
# 보안을 위해 실제 운영 환경에서는 환경 변수나 별도의 설정 파일을 사용하는 것이 좋습니다.
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '0000',
    'database': 'patroldb' # 위에서 생성한 DB 이름
}

def process_and_store_data(data_dict):
    """
    수신한 데이터를 파싱하여 situation과 media 테이블에 트랜잭션으로 저장합니다.
    동영상 프레임 데이터는 이 함수에서 의도적으로 무시합니다.
    
    Args:
        data_dict (dict): 클라이언트로부터 수신한 JSON 데이터 (파이썬 딕셔너리 형태)

    Returns:
        bool: 모든 작업이 성공하면 True, 실패하면 False를 반환합니다.
    """
    connection = None
    try:
        # DB 연결
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # === 트랜잭션 시작 ===
        # 두 개의 INSERT 작업이 모두 성공하거나 모두 실패하도록 보장하여 데이터 무결성을 지킵니다.
        connection.start_transaction()

        # 1. situation 테이블에 데이터 저장
        situation_query = """
            INSERT INTO situation (timestamp, patrol_car, class_name, raw_result)
            VALUES (%s, %s, %s, %s)
        """
        # raw_result 필드는 JSON 타입이므로, 파이썬 딕셔너리를 JSON 문자열로 변환해줍니다.
        raw_result_str = json.dumps(data_dict.get('raw_result'))
        situation_args = (
            data_dict.get('timestamp'),
            data_dict.get('patrol_car'),
            data_dict.get('class_name'),
            raw_result_str
        )
        cursor.execute(situation_query, situation_args)
        
        # 방금 INSERT된 situation 레코드의 기본 키(class_id) 가져오기
        class_id = cursor.lastrowid
        print(f"[DB] 'situation' 테이블에 데이터 저장 완료. 생성된 class_id: {class_id}")

        # 2. media 테이블에 데이터 저장
        media_info = data_dict.get('media_info', {})
        media_query = """
            INSERT INTO media (class_id, rel_path, validation, mime_type)
            VALUES (%s, %s, %s, %s)
        """
        media_args = (
            class_id, # 위에서 얻은 class_id를 외래 키로 사용
            media_info.get('rel_path'),
            media_info.get('validation'),
            media_info.get('mime_type')
        )
        cursor.execute(media_query, media_args)
        print(f"[DB] 'media' 테이블에 데이터 저장 완료.")
        
        # 3. 모든 작업이 성공했으므로 최종적으로 DB에 변경사항을 확정(COMMIT)
        connection.commit()
        # === 트랜잭션 종료 (성공) ===
        
        return True

    except Error as e:
        print(f"[DB] 오류 발생: {e}")
        # 오류 발생 시, 트랜잭션 내의 모든 작업을 취소(ROLLBACK)
        if connection:
            connection.rollback()
            print("[DB] 트랜잭션이 롤백되었습니다.")
        return False
    finally:
        # 성공하든 실패하든 항상 DB 연결을 닫아줍니다.
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            # print("[DB] MySQL 연결이 종료되었습니다.")


def start_server(host='0.0.0.0', port=9999):
    """TCP 서버를 시작하고 클라이언트의 연결을 대기합니다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[서버] {host}:{port} 에서 연결 대기 중...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[서버] 클라이언트 연결됨: {addr}")
                
                # 데이터의 크기가 클 수 있으므로 여러 번에 나눠서 받습니다.
                full_data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    full_data += chunk
                
                if not full_data:
                    print("[서버] 클라이언트로부터 데이터를 받지 못했습니다.")
                    continue

                try:
                    # 수신한 바이트 데이터를 UTF-8 문자열로 디코딩 후, JSON 파싱
                    message = json.loads(full_data.decode('utf-8'))
                    
                    # [요구사항 2] 동영상 프레임은 DB에 저장하지 않으므로,
                    # process_and_store_data 함수를 호출하여 메타데이터만 처리합니다.
                    # 'video_frame' 키의 존재 여부와 상관없이 다른 데이터는 정상 처리됩니다.
                    
                    if process_and_store_data(message):
                        response = {"status": "success", "message": "데이터가 성공적으로 저장되었습니다."}
                    else:
                        response = {"status": "error", "message": "서버 내부 오류로 데이터 저장에 실패했습니다."}

                except json.JSONDecodeError:
                    print("[서버] 오류: 유효하지 않은 JSON 형식입니다.")
                    response = {"status": "error", "message": "Invalid JSON format."}
                except Exception as e:
                    print(f"[서버] 처리 중 오류 발생: {e}")
                    response = {"status": "error", "message": f"An unexpected error occurred: {e}"}

                # 클라이언트에 처리 결과 응답
                conn.sendall(json.dumps(response).encode('utf-8'))

if __name__ == "__main__":
    start_server()
