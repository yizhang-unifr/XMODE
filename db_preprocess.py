import sqlite3
from pathlib import Path
from functools import partial

def get_patient_path_by_subject_id(datasets_path, subject_id):
    datasets_path = Path(__file__).parent / datasets_path
    path = datasets_path.rglob(f'p{subject_id}')
    res = list(path)[0]
    return res
    

# make partial funtion for get_patient_path_by_subject_id
get_image_patient_path_by_subject_id = partial(get_patient_path_by_subject_id, 'files')
get_report_patient_path_by_subject_id = partial(get_patient_path_by_subject_id, 'reports')  

def get_image_path_by_subject_id(subject_id, study_id, image_id):
    res =  get_image_patient_path_by_subject_id(subject_id) 
    if not res:
        return None
    res = res / f's{study_id}' / f'{image_id}.jpg'
    if res.exists():
        return str(res)
    else:
        return None

def get_report_path_by_subject_id(subject_id, study_id):
    res =  get_report_patient_path_by_subject_id(subject_id) 
    if not res:
        return None
    res = res / f's{study_id}.txt'
    if res.exists():
        return str(res)
    else:
        return None

update_image_path_column_command = '''
    UPDATE TB_CXR
    SET image_path = image_patient_path(subject_id, study_id, image_id);
'''

update_report_path_column_command = '''
    UPDATE TB_CXR
    SET report_path = report_patient_path(subject_id, study_id);
'''

def insert_report_path(db_path='mimic_iv_cxr_with_path.db'):
    conn = sqlite3.connect(db_path)
    conn.create_function('report_patient_path', 2, get_report_path_by_subject_id)
    cursor = conn.cursor()
    try:
        cursor.execute(update_report_path_column_command)
        conn.commit()
        print("success")
    except Exception as e:
        print(e)
        conn.rollback()
    finally:
        conn.close()    

def insert_image_path(db_path='mimic_iv_cxr_with_path.db'):
    conn = sqlite3.connect(db_path)
    conn.create_function('image_patient_path', 3, get_image_path_by_subject_id)
    cursor = conn.cursor()
    try:
        cursor.execute(update_image_path_column_command)
        conn.commit()
        print("success")
    except Exception as e:
        print(e)
        conn.rollback()
    finally:
        conn.close()

        
if __name__=='__main__':
    insert_image_path()
    insert_report_path()
    
    



