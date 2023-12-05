import random
import numpy as np
import math
from itertools import product, chain, combinations
from shapely.geometry import LineString, Point, Polygon
from copy import deepcopy
from multiprocessing import Pool


class MACHINE:
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

    def __init__(
            self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]
    ):
        global g_machine
        g_machine = self
        self.id = "MACHINE"
        self.score = [0, 0]  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.board_size = 7  # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = []  # [(a, b), (c, d), (e, f)]
        self.minmaxtree = self.MinMaxNode()
        self.minmax_depth = 3

        self.max_iter = 400

    class MinMaxNode:
        def __init__(self):
            self.value = 0
            self.childs: list[MACHINE.MinMaxNode] = []

            # below should be data of the map and the move last made
            self.drawn_lines = []
            self.lastmove = []  # a line

        def create_childs(self):
            global g_machine
            lines = list(combinations(g_machine.whole_points, 2))
            lines = organise_points(lines)
            for l in lines:
                if not check_avail(l, g_machine.whole_points, self.drawn_lines):
                    continue
                newchild = MACHINE.MinMaxNode()
                newchild.drawn_lines = deepcopy(self.drawn_lines)
                newchild.drawn_lines.append(l)
                newchild.lastmove = l
                self.childs.append(newchild)

            """
            for p in g_machine.whole_points:
                for q in g_machine.whole_points:
                    if p == q:
                        continue
                if (p, q) in self.drawn_lines:
                    continue
                if not check_avail():
                    continue
                newchild = MACHINE.MinMaxNode()
                newchild.drawn_lines = self.drawn_lines
                newchild.lastmove = (p, q)
                self.childs.append(newchild)
                """

        def maximise_child_toplevel(self, depth_count):
            self.create_childs()
            move = ((0, 0), (0, 0))
            self.value = 0
            for c in self.childs:
                ##TODO alphabeta cutoff
                v = c.minimise_child(depth_count - 1, 0)
                if self.value < v:
                    self.value = v
                    move = c.lastmove
            print(f"move was {move}")
            return move

        def maximise_child(self, depth_count, parent_beta):
            # if depth 0 eval.
            if depth_count == 0:
                print(f"bottom, level {depth_count}, move {self.lastmove}")
                return self.eval_node()
            # forall child minimise(depth_count -1)
            print(f"on node depth {depth_count} move {self.lastmove}")
            self.create_childs()
            if len(self.childs) == 0:
                return self.eval_node()  # I don't know why it works...
            for c in self.childs:
                ##TODO alphabeta cutoff
                v = c.minimise_child(depth_count - 1, 0)
                if self.value < v:
                    self.value = v
            print("leaving")
            return v

        def minimise_child(self, depth_count, parent_alpha):
            # if depth 0 eval.
            if depth_count == 0:
                print(f"bottom, level {depth_count}, move {self.lastmove}")
                return self.eval_node()
            # forall child minimise(depth_count -1)
            print(f"on node depth {depth_count} move {self.lastmove}")
            self.create_childs()
            if len(self.childs) == 0:
                return self.eval_node()  # I don't know why it works...
            for c in self.childs:
                ##TODO alphabeta cutoff
                v = c.maximise_child(depth_count - 1, 0)
                if self.value > v:
                    self.value = v
            print("leaving")
            return v

        def eval_node(self):
            return random.randrange(0, 10)
            # return 0
            # check_triangle()

    def find_best_selection(self):
        # self.minmaxtree.drawn_lines = deepcopy(self.drawn_lines)
        # return list(self.minmaxtree.maximise_child_toplevel(self.minmax_depth))

        if len(self.drawn_lines) <= len(self.whole_points)/2:
            return self.rule_based_selection()

        else:

            (depth, childs) = self.determine_depth()
            ex = -100
            line = [(0, 0), (0, 0)]
            ran = len(self.whole_points) / 4
            ran = int(ran)
            p = Pool(6)
            ret1 = p.apply_async(self.max, (-100, 100, depth, depth, childs, ran, 0))
            ret2 = p.apply_async(self.max, (-100, 100, depth, depth, childs, ran, 1))
            ret3 = p.apply_async(self.max, (-100, 100, depth, depth, childs, ran, 2))
            ret4 = p.apply_async(self.max, (-100, 100, depth, depth, childs, ran, 3))
            (t_ex, t_line) = ret1.get()
            print(t_ex, t_line)
            if t_ex > ex:
                ex = t_ex
                line = t_line
            (t_ex, t_line) = ret2.get()
            print(t_ex, t_line)
            if t_ex > ex:
                ex = t_ex
                line = t_line
            (t_ex, t_line) = ret3.get()
            print(t_ex, t_line)
            if t_ex > ex:
                ex = t_ex
                line = t_line
            (t_ex, t_line) = ret4.get()
            print(t_ex, t_line)
            if t_ex > ex:
                ex = t_ex
                line = t_line
            return line

            # (depth, childs)=self.determine_depth()
            # (ex, line) = self.max(-2,2,depth, childs)
            # return line

    # (depth, childs)
    def determine_depth(self):
        count = 0
        """
        lines = list(combinations(self.whole_points, 2))
        lines = organise_points(lines)
        for l in lines:
            if not self.check_availability(l):
                continue
            count+=1
        """
        count = len([[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                     self.check_availability([point1, point2])])
        # real endgame - all child node can be evaluated
        if (math.factorial(count) <= self.max_iter):
            return (32767, 32767)
        # near endgame - possibly small enough that depth might reach bottom
        available_whole_lines = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                                 self.count_valid_lines([point1, point2])]
        if (len(self.drawn_lines) > (1 / 2) * len(available_whole_lines)):
            for i in range(10):
                if (count ** i > self.max_iter):
                    return (i - 1, 32767)
        # midgame - meaningless, just run pre-known value
        # TODO make a check logic to determine it
        for i in range(10):
            if (5 ** i > self.max_iter):
                return (i - 1, 5)

    def select_promising(self, lines, c=1):
        return random.sample(lines, c)

    def max(self, alpha, beta, depth, top, cnodes, ran, id):
        maxv = -100
        max_line = [(0, 0), (0, 0)]

        if self.check_endgame() or depth == 0:
            maxv = self.score[1] - self.score[0]
            print(self.drawn_lines)
            print(maxv)
            return (maxv, max_line)

        # 원본코드 보존을 위해 조건을 이렇게 넣었습니다. 괜찮다면 나중에 이 버전으로 전부 돌리세요.
        """if(cnodes!=32767):
            child_nodes=[[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
            for line in self.select_promising(child_nodes):
                self.drawn_lines.append(line)
                tf = self.evaluate(line, 1)
                (m, min_line) = self.min(alpha, beta, depth - 1, cnodes)
                if (m > maxv):
                    maxv = m
                    max_line = line
                self.drawn_lines.remove(line)
                if tf:
                    self.score[1]-=1
                    self.triangles.pop()

                if maxv >= beta:
                    return (maxv, max_line)

                if maxv > alpha:
                    alpha = maxv
            return (maxv, max_line)
        """

        if depth == top:
            for i in range(id * ran, ((id + 1) * ran) if id != 3 else len(self.whole_points)):
                for j in range(0, len(self.whole_points)):
                    if self.whole_points[i] == self.whole_points[j]: continue
                    if self.check_availability([self.whole_points[i], self.whole_points[j]]):
                        line = self.organize_points([self.whole_points[i], self.whole_points[j]])
                        self.drawn_lines.append(line)
                        tf = self.evaluate(line, 1)
                        (m, min_line) = self.min(alpha, beta, depth - 1, top, cnodes)
                        if (m != 100 and m > maxv):
                            maxv = m
                            max_line = line
                        self.drawn_lines.remove(line)
                        if tf:
                            self.score[1] -= 1
                            self.triangles.pop()

                        if maxv >= beta:
                            return (maxv, max_line)

                        if maxv > alpha:
                            alpha = maxv
            return (maxv, max_line)
        else:
            for i in range(0, len(self.whole_points)):
                for j in range(i, len(self.whole_points)):
                    if self.whole_points[i] == self.whole_points[j]: continue
                    if self.check_availability([self.whole_points[i], self.whole_points[j]]):
                        line = self.organize_points([self.whole_points[i], self.whole_points[j]])
                        self.drawn_lines.append(line)
                        tf = self.evaluate(line, 1)
                        (m, min_line) = self.min(alpha, beta, depth - 1, top, cnodes)
                        if (m != 100 and m > maxv):
                            maxv = m
                            max_line = line
                        self.drawn_lines.remove(line)
                        if tf:
                            self.score[1] -= 1
                            self.triangles.pop()

                        if maxv >= beta:
                            return (maxv, max_line)

                        if maxv > alpha:
                            alpha = maxv
            return (maxv, max_line)

    def min(self, alpha, beta, depth, top, cnodes):
        minv = 100
        min_line = [(0, 0), (0, 0)]

        if self.check_endgame() or depth == 0:
            minv = self.score[1] - self.score[0]
            print(self.drawn_lines)
            print(minv)
            return (minv, min_line)

        # 원본코드 보존을 위해 조건을 이렇게 넣었습니다. 괜찮다면 나중에 이 버전으로 전부 돌리세요.
        """if(cnodes!=32767):
            child_nodes=[[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
            for line in self.select_promising(child_nodes):
                self.drawn_lines.append(line)
                tf = self.evaluate(line, 0)
                (m, max_line) = self.max(alpha, beta, depth - 1, cnodes)
                if (m < minv):
                    minv = m
                    min_line = line
                self.drawn_lines.remove(line)
                if tf:
                    self.score[0]-=1
                    self.triangles.pop()

                if minv <= alpha:
                    return (minv, min_line)

                if minv < beta:
                    beta = minv
            return (minv, min_line)
        """

        for i in range(0, len(self.whole_points)):
            for j in range(i, len(self.whole_points)):
                if self.whole_points[i] == self.whole_points[j]: continue
                if self.check_availability([self.whole_points[i], self.whole_points[j]]):
                    line = self.organize_points([self.whole_points[i], self.whole_points[j]])
                    self.drawn_lines.append(line)
                    tf = self.evaluate(line, 0)
                    (m, max_line) = self.max(alpha, beta, depth - 1, top, cnodes, 0, 0)
                    if (m != -100 and m < minv):
                        minv = m
                        min_line = line
                    self.drawn_lines.remove(line)
                    if tf:
                        self.score[0] -= 1
                        self.triangles.pop()

                    if minv <= alpha:
                        return (minv, min_line)

                    if minv < beta:
                        beta = minv
        return (minv, min_line)

    def organize_points(self, point_list):
        point_list.sort(
            key=lambda x: (x[0], x[1])
        )  # x[0],x[1]을 기준으로 정렬해서 저장, 즉 x값을 기준으로 오름차순으로 정렬하고 x값이 같다면 y값을 기준으로 정렬
        return point_list

    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)

        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point == line[0] or point == line[1]:
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

    def rule_based_selection(self):  # return : [(x1, y1), (x2, y2)]

        # 1. 이미 완성된 사각형이 있다면 그 사이를 잇는 선분을 그리기
        rects = self.find_rectangles()
        drawn_lines = self.drawn_lines
        for rectangle in rects:  # 사각형이 있다면(내부에 점 없고, 삼각형1+선분1 조합 아니고, )
            for candi in list(combinations(rectangle, 2)):  # candi : 점 2개 조합
                # print("candi : ", candi)
                if (candi not in drawn_lines) and (self.check_availability(candi)) and not (
                self.if_find_triangles(candi)):  # true라면
                    print("1. 이미 완성된 사각형에 긋기 : ", list(candi))
                    return list(candi)

        # heuristic #2 : 한 점에서 이미 두 선분이 이어졌다면 그 두 선분을 이어야 한다(by jiwon)
        points_to_connect = self.find_candidate()  # array
        # print("points_to_connect : ", points_to_connect)
        if points_to_connect:
            # 3번 이상 연결된 점이 있는 경우 상대 점들을 둘씩 조합하여 가능성을 따짐
            for combi in points_to_connect:
                if len(combi) > 3:
                    # print("조합 가능성 combi[1:] : ", combi[1:])
                    for [pointA, pointB] in list(combinations(combi[1:], 2)):
                        # print("후보 삼각형 : ", pointA, pointB, combi[0])
                        if self.check_triangle([pointA, pointB], 1):
                            # if self.probability_make_Triangle(pointA, pointB, combi[0]):
                            # print("poin1, point2 : ", pointA, pointB)
                            if self.check_availability([pointA, pointB]):
                                print("2 : 한 점에서 이미 두 선분이 이어졌다면 : ", [pointA, pointB])
                                return [pointA, pointB]
                            else:
                                pass

                # 2번만 연결되었다면 상대 두 점을 그대로 반환
                elif len(combi) == 3:
                    # print("combi :", combi)
                    if self.check_availability(combi[1:]):
                        if self.check_triangle(combi[1:], 1):
                            # if (self.probability_make_Triangle(combi[0], combi[1], combi[2])):
                            print("3. 삼각형의 두 선분이 완성된 경우 : ", combi[1:])
                            return combi[1:]
                # 2번 이상 연결된 점이 1개 이하라면 pass
                else:
                    pass

        # 상대방이 만든 점이 포함된 삼각형을 찾아 연결
        empty_triangles = self.find_triangles(self.drawn_lines)
        point_count = self.count_connected_lines()

        # 빈 삼각형 안에 존재하는 점 찾기
        if len(empty_triangles) > 0:
            for point in self.whole_points:

                if Polygon(empty_triangles).intersection(Point(point)):
                    # Polygon(Point(point)).within(empty_triangles): # 빈 삼각형 안에 위치한 점에 대해서
                    count = 0
                    # 가운데 위치한 점이 연결된 횟수 가져오기
                    for row in point_count:
                        if row[0] == point:
                            count = row[1]
                        # 빈 삼각형 안에 존재하는 점이 연결된 횟수가 짝수라면 연결
                        if count % 2 == 0:
                            for vertex in empty_triangles:
                                if self.check_availability([vertex, point]):
                                    print("4. 빈 삼각형 - 짝수 : ", [vertex, point])
                                    return [vertex, point]
                        # 빈 삼각형 안에 존재하는 점이 연결된 횟수가 홀수라면 연결x

        # 아무 선분도 연결되지 않은 두 점 찾기
        unconnected_points = []

        for point in self.whole_points:
            connected = False

            for line in self.drawn_lines:
                if point in line:
                    connected = True
                    continue

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
            new_choice = list(random.choice(available_new_lines))
            print("5. 연결되지 않은 두 점 연결 : ", new_choice)
            return new_choice

        # heuristic #5 : 휴리스틱으로 골라낼 수 있는 선분이 없다면 랜덤으로 선택(by jiwon)
        # -> 기존 find_best_selection 함수 그대로
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                     self.check_availability([point1, point2])]
        choice = random.choice(available)
        print("6. random : ", choice)
        return choice

    def find_rectangles(self):
        drawn_lines = self.drawn_lines
        rectangles = []  # 사각형 list
        if (len(drawn_lines) > 3):
            for combi in list(combinations(drawn_lines, 4)):
                # print("combi in find_rectangles: ", combi)
                points = set()
                for pointA, pointB in combi:
                    points.add(pointA)
                    points.add(pointB)
                if len(points) == 4:
                    # TODO. if 그 안에 삼각형이 없다면 (삼각형1+선분1) 조합이 아닌지 확인 -> 나머지 선분을 잇기는 해서 문제 x
                    rectangles.append(points)
        # print("find_rectangels 결과 : ", rectangles)
        return rectangles

    # 색칠되지 않는 삼각형을 찾는 함수
    def find_triangles(self, drawn_lines):
        empty_triangles = []  # 빈 삼각형 list
        result = []
        # print("기존 삼각형 : ", self.triangles)
        if (len(drawn_lines) > 2):
            for combi in list(combinations(drawn_lines, 3)):
                # print("combi : ", combi)
                points = set()
                for pointA, pointB in combi:
                    points.add(pointA)
                    points.add(pointB)

                if len(points) == 3:
                    # print("삼각형 확인중인 점들 points : ", points)
                    # 색칠된 삼각형은 제외
                    for triangle in self.triangles:
                        if points not in triangle:
                            empty_triangles.append(points)
            result = [list(subset) for subset in empty_triangles]
            # print("그어진 선분이 3개 이상인 경우 empty_triangle :", result)

        # print("empty_triangle :", result)
        if len(result) > 0:
            # print("return_empty_triangle(result[0]) :", result[0])
            return result[0]
        else:
            # print("empty_triangle :", result)
            return result

    # 이 선분을 그으면 점을 포함하는 삼각형이 생성될까?
    # true면 생성된다는 의미
    def if_find_triangles(self, line):
        drawn_lines = self.drawn_lines
        drawn_lines.append(line)  # 이 선분을 그었을 때 삼각형 안에 점이 생성될지를 검사하고자 함.

        result = self.find_triangles(drawn_lines)

        # print("if_empty_triangle :", result)
        if len(result) > 0:
            # print("if_empty_triangle(result[0]) :", result[0])
            return True
        else:
            return False

    # 각 점에서 연결된 선분의 개수를 세는 함수
    #  -> TODO. 한번도 연결되지 않은 선분을 찾을 때 사용하지 않을 거라면 count_connected_lines_two로 합칠 것
    def count_connected_lines(self):
        whole_points = self.whole_points
        drawn_lines = self.drawn_lines
        count_connected = []  # 2차원 배열. [[(x좌표, y좌표), 각 점이 다른 점들과 연결된 횟수], [], ... ]

        # 2차원 배열의 1열에 whole_points 점 각각 넣음
        index = 0
        for point in whole_points:
            count_connected.append([])
            count_connected[index].append(point)
            index += 1
        # print("count_connected : ", count_connected)

        # 2차원 배열의 2열에 각 점이 다른 점들과 연결된 횟수 넣음
        index = 0
        for point in whole_points:
            count = 0
            for line in drawn_lines:
                if line[0] == point or line[1] == point:
                    count += 1
            else:
                pass
            count_connected[index].append(count)
            index += 1
        return count_connected

    # 2번 이상 선택된 점들 찾는 함수
    def points_count_two(self):
        count_all_points = self.count_connected_lines()  # 2차원 배열. [[(x좌표, y좌표), 각 점이 다른 점들과 연결된 횟수], [], ... ]

        point_selected_2times = []  # 2번 이상 선택된 점들
        # print("count_all_points :", count_all_points)
        for set in count_all_points:
            if set[1] >= 2:
                point_selected_2times.append(set[0])  # TODO. 두번 연결된 점이 여러 개인 경우 어떻게 할지 결정해야 함
        # print("point_selected_2times :", point_selected_2times)

        return point_selected_2times

    # 2번 이상 연결된 점과 그에 연결된 점 [[2번이상 연결된 점, 그에 연결된 점 1, 그에 연결된 점 2], ...]
    # 연결 후보 : [][1], [][2]
    def find_candidate(self):
        selected_2times = self.points_count_two()
        drawn_lines = self.drawn_lines
        points_to_connect = []  # 연결 후보

        # 두 번 이상 연결된 점에 연결된 점들 반환(그 점들을 이으면 삼각형이 완성될 가능성이 있는)
        index = 0
        for point in selected_2times:
            points_to_connect.append([])
            points_to_connect[index].append(point)
            for line in drawn_lines:
                if line[0] == point:
                    points_to_connect[index].append(line[1])
                elif line[1] == point:
                    points_to_connect[index].append(line[0])
                else:
                    pass
            index += 1

        # print("points_to_connect :", points_to_connect)
        return points_to_connect

    '''사용 x'''

    # 만들어질 삼각형 안에 점이 존재하는지 확인. 존재하는 경우 false 반환 -> true인 경우 그어도 됨.
    def probability_make_Triangle(self, point1, point2, point3):
        triangle = [point1, point2, point3]
        for point in self.whole_points:
            if bool(Polygon(triangle).intersection(Point(point))):  # 점이 없음
                return True
            else:
                False  # 점이 없음

    def evaluate(self, line, turn):
        tf = self.check_triangle(line, turn)
        return tf

    def check_endgame(self):
        remain_to_draw = [
            [point1, point2]
            for (point1, point2) in list(combinations(self.whole_points, 2))
            if self.check_availability([point1, point2])
        ]
        return False if remain_to_draw else True

    def check_triangle(self, line, turn):
        point1 = line[0]
        point2 = line[1]

        point1_connected = []
        point2_connected = []

        for l in self.drawn_lines:  # 그려진 선들을 살펴보면서 나 자신을 제외한 선분들이 어떤 점과 연결되어 있는지를 저장
            if l == line:  # 자기 자신 제외
                continue
            if point1 in l:
                point1_connected.append(l)
            if point2 in l:
                point2_connected.append(l)

        if point1_connected and point2_connected:  # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
            for line1, line2 in product(point1_connected, point2_connected):
                # Check if it is a triangle & Skip the triangle has occupied
                triangle = self.organize_points(list(set(chain(*[line, line1, line2]))))
                if len(triangle) != 3 or triangle in self.triangles:
                    continue

                empty = True
                for point in self.whole_points:
                    if point in triangle:
                        continue
                    polygon = Polygon(triangle)
                    if polygon.is_valid:  # Check if the polygon is valid
                        if bool(polygon.intersection(Point(point))):  # Check if the point is inside the polygon
                            empty = False

                if empty:
                    self.score[turn] += 1
                    self.triangles.append(triangle)
                    return 1
        return 0

    def count_valid_lines(self, line):

        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)

        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point == line[0] or point == line[1]:
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

        if condition1 and condition2 and condition3:
            return True
        else:
            return False


def organise_points(point_list):
    point_list.sort(
        key=lambda x: (x[0], x[1])
    )  # x[0],x[1]을 기준으로 정렬해서 저장, 즉 x값을 기준으로 오름차순으로 정렬하고 x값이 같다면 y값을 기준으로 정렬
    return point_list


##function version of checkavail
def check_avail(line, whole_points, drawn_lines):
    line_string = LineString(line)

    # Must be one of the whole points
    condition1 = (line[0] in whole_points) and (line[1] in whole_points)

    # Must not skip a dot
    condition2 = True
    for point in whole_points:
        if point == line[0] or point == line[1]:
            continue
        else:
            if bool(line_string.intersection(Point(point))):
                condition2 = False

    # Must not cross another line
    condition3 = True
    for l in drawn_lines:
        if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
            continue
        elif bool(line_string.intersection(LineString(l))):
            condition3 = False

    # Must be a new line
    condition4 = line not in drawn_lines

    condition5 = line[0] != line[1]

    if condition1 and condition2 and condition3 and condition4 and condition5:
        return True
    else:
        return False


g_machine: MACHINE = 0