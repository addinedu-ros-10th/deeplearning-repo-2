# server/data_service.py

import json
from datetime import datetime
import db_connector
import os
from config import MEDIA_STORAGE_ROOT

def get_logs(start_date, end_date, event_types=None, orderby_latest=True):
    conn = db_connector.connect_db()
    if not conn: return []
    logs = []
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT class_id, timestamp, raw_result FROM situation WHERE DATE(timestamp) BETWEEN %s AND %s"
        params = [start_date, end_date]
        if event_types:
            event_conditions = ["raw_result LIKE %s" for _ in event_types]
            params.extend([f'%"class_name": "{event}"%' for event in event_types])
            query += f" AND ({' OR '.join(event_conditions)})"
        order_clause = "ORDER BY timestamp DESC" if orderby_latest else "ORDER BY timestamp ASC"
        query += " " + order_clause
        cursor.execute(query, tuple(params))
        logs = cursor.fetchall()
    except Exception as e:
        print(f"데이터 조회 중 에러 발생: {e}")
    finally:
        cursor.close()
        conn.close()
    return logs

def get_video_path_for_log(class_id):
    conn = db_connector.connect_db()
    if not conn: return None
    path = None
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT rel_path FROM media WHERE class_id = %s LIMIT 1"
        cursor.execute(query, (class_id,))
        result = cursor.fetchone()
        if result and result['rel_path']:
            full_path = os.path.abspath(os.path.join(MEDIA_STORAGE_ROOT, result['rel_path']))
            path = full_path
    except Exception as e:
        print(f"영상 경로 조회 중 에러 발생: {e}")
    finally:
        cursor.close()
        conn.close()
    return path