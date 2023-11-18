import random
from itertools import combinations
from shapely.geometry import LineString, Point
from copy import deepcopy

##remove below imports when it is not needed by chekc-tris
from itertools import product, chain, combinations
from shapely.geometry import LineString, Point, Polygon


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
        # TODO get changed game state and change root of node to opponents move
        # but is that necessasary...?
        self.minmaxtree.drawn_lines = deepcopy(self.drawn_lines)
        return list(self.minmaxtree.maximise_child_toplevel(self.minmax_depth))

        available = [
            [point1, point2]
            for (point1, point2) in list(combinations(self.whole_points, 2))
            if self.check_availability([point1, point2])
        ]
        return random.choice(available)

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
        condition4 = line not in self.drawn_lines

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False


g_machine: MACHINE = 0


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


##copied form system.py
def organise_points(point_list):
    point_list.sort(key=lambda x: (x[0], x[1]))
    return point_list


##copied from system.py
def check_triangle(drawn_lines, my_tris, en_tris, line):
    point1 = line[0]
    point2 = line[1]

    point1_connected = []
    point2_connected = []

    for l in drawn_lines:
        if l == line:  # 자기 자신 제외
            continue
        if point1 in l:
            point1_connected.append(l)
        if point2 in l:
            point2_connected.append(l)

    if point1_connected and point2_connected:  # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
        for line1, line2 in product(point1_connected, point2_connected):
            # Check if it is a triangle & Skip the triangle has occupied
            triangle = organize_points(list(set(chain(*[line, line1, line2]))))
            if len(triangle) != 3 or triangle in self.triangles:
                continue

            empty = True
            for point in self.whole_points:
                if point in triangle:
                    continue
                if bool(Polygon(triangle).intersection(Point(point))):
                    empty = False

            if empty:
                self.triangles.append(triangle)
                self.score[PLAYERS.index(self.turn)] += 1

                color = USER_COLOR if self.turn == "USER" else MACHINE_COLOR
                self.occupy_triangle(triangle, color=color)
                self.get_score = True
