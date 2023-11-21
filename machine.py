import random
from itertools import combinations
from shapely.geometry import LineString, Point

class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """
    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]

    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)
    
    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)
        
        # TODO : 그릴 수 있는 선에 대한 조건

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    
        
    
    # rule 기반 알고리즘  
    def rule_based_selection(self):
        
        '''
        # 1. 이미 완성된 사각형이 있다면 그 사이를 잇는 선분을 그리기
        if(완성된 사각형이 있는지 확인하는 함수)  #TODO. 사각형이 있는지 확인하는 함수
            # TODO. 이미 완성된 사각형이 있다면 그 사이를 잇는 선분 그리기
            return 그 사이를 잇는 선분
        ''' 

        # 아무 선분도 연결되지 않은 두 점 찾기
        unconnected_points = []

        for point in self.whole_points:
            connected = False

            for line in self.drawn_lines:
                if point in line:
                    connected = True
                    break  # 하나의 연결된 선분을 찾았으면 더 이상 확인할 필요가 없으므로 반복문 종료

            if not connected:
                unconnected_points.append(point)

        # unconnected_points 중 두 점을 연결해서 선분 만들기
        new_lines = []

        for i in range(len(unconnected_points) - 1):
            for j in range(i + 1, len(unconnected_points)):
                point1 = unconnected_points[i]
                point2 = unconnected_points[j]
                new_line = (point1, point2)
                new_lines.append(new_line)

        # 가능한 선분인지 확인
        available_new_lines = []

        for new_line in new_lines:
            if self.check_availability(new_line):
                available_new_lines.append(new_line)
                
        if available_new_lines:
            return random.choice(available_new_lines)
    
        # by jiwon(heuristic #2 : 한 점에서 이미 두 선분이 이어졌다면 그 두 선분을 이어야 한다)
        points_to_connect = self.count_connected_lines_two()
        if points_to_connect:

            # 3번이상 연결된 점이 있는 경우 둘씩 조합하여 가능성을 따짐
            if len(points_to_connect) > 2:
                combinations_points = list(combinations(points_to_connect, 2))
                for line in combinations_points:
                    if self.check_availability(line): # TODO. 해당 선분을 이어 삼각형이 생성될 경우 그 삼각형 안에 점이 없는지 확인하는 조건을 추가해야 함
                        return line

            # 2번만 연결되었다면 상대 두 점을 그대로 반환
            elif len(points_to_connect)== 2:
                return points_to_connect
            
            # 2번 이상 연결된 점이 1개 이하라면 pass
            else:
                pass

        # by jiwon(heuristic #5 : 휴리스틱으로 골라낼 수 있는 선분이 없다면 랜덤으로 선택)
        # 기존 find_best_selection 함수 그대로
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)

    # 각 점에서 연결된 선분의 개수를 세는 함수
    #  -> TODO. 한번도 연결되지 않은 선분을 찾을 때 사용하지 않을 거라면 count_connected_lines_two로 합칠 것
    def count_connected_lines(self):
        whole_points = self.whole_points
        drawn_lines = self.drawn_lines
        count_connected = []
        point_index =0

        for point in whole_points:
            if point in drawn_lines:
                for line in drawn_lines:
                    if line[0] == point or line[1] == point:
                        count_connected[point_index] += 1
                point_index += 1
            else:
                count_connected[point_index] = 0

        return count_connected
    
    # 2번 이상 선택된 점과 그에 연결된 점 
    def count_connected_lines_two(self):
        count_all_points = self.count_connected_lines()
        drawn_lines = self.drawn_lines

        #points_selected_2times = []
        points_to_connect = [] # 연결 후보

        index = 0

        # 두번 이상 연결된 점 찾기
        for c in count_all_points:
            if c >= 2:
                point_selected_2times = count_all_points[index] #TODO. 두번 연결된 점이 여러개인 경우 어떻게 할지 결정해야 함
                break
            index += 1

        # 두 번 이상 연결된 점에 연결된 점들 반환(그 점들을 이으면 삼각형이 완성될 가능성이 있는)
        for line in drawn_lines:
            if line[0] == point_selected_2times:
                points_to_connect.append(line[1])
            elif line[1] == point_selected_2times:
                points_to_connect.append(line[0])
            else:
                pass

        return points_to_connect
