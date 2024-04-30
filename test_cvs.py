import csv

# def get_diagnosis_from_id(csv_file, id_code):
#     with open(csv_file, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             if row['id_code'] == id_code:
#                 return row['diagnosis']
#     return None

# # CSV 파일 경로
# csv_file = '/home/gyutae/atops2019/train_1.csv'

# # 확인할 id_code
# id_code_to_check = '1ae8c165fd53'

# # 해당 id_code의 diagnosis 가져오기
# diagnosis = get_diagnosis_from_id(csv_file, id_code_to_check)
# if diagnosis is not None:
#     print(f"The diagnosis for id_code {id_code_to_check} is: {diagnosis}")
# else:
#     print(f"No diagnosis found for id_code {id_code_to_check}")
import pandas as pd

# CSV 파일을 읽어옵니다.
file_path = '/home/gyutae/atops2019/test.csv'  # 파일 경로를 적절히 지정해주세요.
data = pd.read_csv(file_path)

# 클래스 개수를 세고자 하는 컬럼 이름을 지정합니다.
target_column = 'your_target_column'  # 클래스를 포함한 컬럼 이름을 입력해주세요.

# 지정된 컬럼에서 클래스 개수를 세고 결과를 출력합니다.
class_counts = data["diagnosis"].value_counts()
print(class_counts)
