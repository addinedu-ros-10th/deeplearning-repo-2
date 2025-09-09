import yaml
from transitions import Machine # 순찰로봇 상태 관리

class FireManager:
  def __init__(self, config_path):
    # 1. YAML 로드
    with open(config_path, 'r', encoding = 'utf-8') as f:
      config = yaml.safe_load(f)
    
    # 2. 파라미터 초기화 (연기/화재 등 갯수)
    for param in config['param']:
      setattr(self, param['name'], param['initial_value'])
    
    # 3. 상태 기계(Machine) 초기화
    self.machine = Machine(
      model = self,
      states=[s['name'] for s in config['state']],
      transitions = config['transitions'],  # 상태 전이(변경) 변수 전달
      initial='NORMAL'  # 초기상태 설정 : 정상
    )
  
  def update_sensor_counts(self, fire_count = None, 
                            danger_smoke_count = None,
                            general_smoke_count = None,
                            irrivent_count = None):
    """화재, 연기 감지값을 받아 상태 값을 업데이트"""
    if fire_count is not None:
      self.fire_count = fire_count
    if danger_smoke_count is not None:
      self.danger_smoke_count = danger_smoke_count
    if general_smoke_count is not None:
      self.general_smoke_count = general_smoke_count
    if irrivent_count is not None:
      self.irrivent_count = irrivent_count

    print(f"감지 값 업데이트: [화재:{self.fire_count}],[화재연기:{self.danger_smoke_count}],[일반연기:{self.general_smoke_count}],[일반객체:{self.irrivent_count}]")
    # print(f"이전 상태: {self.state}")

    # ==========================================================
    # 조건(Condition) 로직이 Python 코드로 이동된 부분
    # ==========================================================
    # 현재 객체(self)의 상태와 변수 값을 보고 어떤 trigger를 호출할지 결정
    
    """
    transitions라이브러리는 yaml파일을 읽어와 self.state, self.to_fire() 등을 
    자동으로 추가하므로 별도 구현이 필요하지 않음
    """

    # 화재/화재연기 1개 이상 감지시 경보(ALERT) 상태로 변경
    if self.fire_count >= 1 or self.danger_smoke_count >= 1:
      if self.state != 'ALERT':
        self.to_alert() # 'to_fire' 트리거 호출

    # 일반연기 1개 이상 감지시 경고(WARNING) 상태로 변경    
    if self.general_smoke_count >= 1:
      if self.state != 'WARNING':
        self.to_warning() # 'to_fire' 트리거 호출
    
    # 화재/화재연기,일반연기가 모두 0개일 경우에 정상(NORMAL) 상태로 변경
    if self.fire_count < 1 and self.danger_smoke_count < 1 and self.general_smoke_count < 1:
      if self.state != 'NORMAL':
        self.to_normal() # 'to_fire' 트리거 호출
    
    print(f"현재 상태: {self.state}")


def main():
  """
  FireManager 데모
  """
  # 1. 초기화
  fm = FireManager("fire_state.yaml")
  
  # 2. 센서값 업데이트 - 아무것도 감지되지 않음
  fm.update_sensor_counts(0,0,0,0)
  
  # 3. 센서값 업데이트 - 일반연기 1개 감지
  fm.update_sensor_counts(0,0,1,0)

  # 4. 센서값 업데이트 - 화재연기 1개 감지
  fm.update_sensor_counts(0,1,0,0)
  
  # 5. 센서값 업데이트 - 아무것도 감지되지 않음
  fm.update_sensor_counts(0,0,0,0)
  
if __name__ == '__main__':
  main()