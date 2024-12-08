import numpy as np
from numpy import ndarray
import warnings

class Point:
    def __init__(self, x, y):
        if not (np.isscalar(x) and np.isscalar(y)):
            raise TypeError("The arguments x and y must be scalar numbers supported by NumPy.")

        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"
    
class Point3d:
    def __init__(self, x, y, z):
        if not (np.isscalar(x) and np.isscalar(y) and np.isscalar(z)):
            raise TypeError("The arguments x, y and z must be scalar numbers supported by NumPy.")

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
class ArcOfCircle:
    def __init__(self, begin_point: Point, center_point: Point, end_point: Point):
        self._validate_point(begin_point, "begin_point")
        self._validate_point(end_point, "end_point")
        self._validate_point(center_point, "center_point")

        self.begin_point = begin_point
        self.end_point = end_point
        self.center_point = center_point

    def _validate_point(self, point, point_name):
        if not isinstance(point, Point):
            raise TypeError(f"{point_name} should be an instance of Point")

    def __repr__(self):
        return f"Arc({self.begin_point}, {self.center_point}, {self.end_point})"

    def radius(self):
        radius_begin = np.sqrt((self.begin_point.x - self.center_point.x)**2 + (self.begin_point.y - self.center_point.y)**2)

        radius_end = np.sqrt((self.end_point.x - self.center_point.x)**2 + (self.end_point.y - self.center_point.y)**2)

        if radius_begin == radius_end:
            return radius_begin
        #else:
            #raise ValueError("Object should be an arc of circle with equal x and y radii.")
        
    def begin_angle(self):
        angle = (np.arctan2(self.begin_point.y, self.begin_point.x) + 2 * np.pi) % (2 * np.pi)
        return angle
    
    def end_angle(self):
        angle = (np.arctan2(self.end_point.y, self.end_point.x) + 2 * np.pi) % (2 * np.pi)
        return angle
    
    def angle_in_between(self):
        #só retorna valores positivos
        angle = self.end_angle() - self.begin_angle()
        if angle < 0.0:
            return (np.pi*2) + angle
        else:
            return angle
        
    def hiperplane(self) -> Point:
        # Valores em radianos
        abs_angle = self.angle_in_between()

        p_med_x = np.cos((abs_angle / 2)+self.begin_angle())
        p_med_y = np.sin((abs_angle / 2)+self.begin_angle())

        arc_len =  abs_angle # raio = 1
        
        hiperplane_x = (1 - (arc_len) / (np.pi * 2)) * p_med_x
        hiperplane_y = (1 - (arc_len) / (np.pi * 2)) * p_med_y

        return Point(hiperplane_x, hiperplane_y)

class Interval:
    def __init__(self, a: Point, b: Point):
        '''
        Interval[a, b]
        '''
        self.a = a
        self.b = b

    def __repr__(self):
        return f"Interval({self.a}, {self.b})"
    
    def hiperplane(self):
        return Point(self.a.x, self.b.x)
    