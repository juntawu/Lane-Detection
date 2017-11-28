

def calculate_cross(two_lines):
    try:
        x1 = two_lines[0,0]
        y1 = two_lines[0,1]
        x2 = two_lines[0,2]
        y2 = two_lines[0,3]
        x3 = two_lines[1,0]
        y3 = two_lines[1,1]
        x4 = two_lines[1,2]
        y4 = two_lines[1,3]
        dx = (x4-x3)*(y2-y1)-(x2-x1)*(y4-y3)
        dy = -dx
        if dx == 0:
            # print('两直线平行，无交叉点')
            return None
        else:
            x = (  (y3-y1)*(x2-x1)*(x4-x3) - x3*(x2-x1)*(y4-y3) + x1*(x4-x3)*(y2-y1)  ) / dx
            y = (  (x3-x1)*(y2-y1)*(y4-y3) - y3*(x4-x3)*(y2-y1) + y1*(x2-x1)*(y4-y3)  ) / dy
        return [int(x), int(y)]
    except:
        return None