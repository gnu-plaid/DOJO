import numpy
import numpy as np
'''
utils for math tools used in creating env
'''

def rotate_vector(vector: np.array([float, float]), angle: int) -> np.array:
    """
    :param vector: Input 2-D vector
    :param angle: rotating angle, IN DEGREE
    :return: rotated vector
    """
    ang2rad = np.deg2rad(angle)
    r_mat = numpy.array([[np.cos(ang2rad), np.sin(ang2rad)],
                         [-np.sin(ang2rad), np.cos(ang2rad)]])
    post_vector = (r_mat.dot(vector)).astype('float')

    return post_vector


def angle_between(angle_base: int, angle: int) -> int:
    """
    :param angle_base: Input angle A
    :param angle: Input angle B
    :return: angle A-B normalize to [-180,180)
    """
    return ((angle_base - angle) + 180) % 360 - 180


def distance_between(dot1: np.array, dot2: np.array) -> float:
    """
    calculate distance between 2 dots
    :param dot1:
    :param dot2:
    :return: distance
    """
    return np.sqrt(np.sum(np.square(dot1 - dot2))).astype('float')

def cross_product(vector1: np.array([float, float]), vector2: np.array([float, float])) -> float:
    """
    used in determine clockwise
    :param vector1:
    :param vector2:
    :return: cross product of 2-d vector
    """
    return vector1[0] * vector2[1] - vector2[0] * vector1[1]

def if_cross(vector1: np.array([[float, float], [float, float]]),
             vector2: np.array([[float, float], [float, float]])) -> bool:
    """
    check if vector1 cross with vector 2
    construct the auxiliary vector and use the vector cross product to determine whether clockwise
    :param vector1:
    :param vector2:
    :return: if cross -> bool
    """
    # auxiliary vector
    vector_1s_2s = vector1[0] - vector2[0]
    vector_1s_2e = vector1[0] - vector2[1]
    vector_1s_1e = vector1[0] - vector1[1]
    return True if cross_product(vector_1s_2s, vector_1s_1e) * cross_product(vector_1s_2e, vector_1s_1e) < 0 else False

def if_straddle(vector1: np.array([[float, float], [float, float]]),
                vector2: np.array([[float, float], [float, float]])) -> bool:
    """
    return if 2 vector cross with each other
    :param vector1:
    :param vector2:
    :return: if straddle -> bool
    """
    return if_cross(vector1, vector2) and if_cross(vector2, vector1)

def is_vector_opposite(vector1: np.array([float, float]),
                       vector2: np.array([float, float]),
                       threshold: int = 170) -> bool:
    """
    check if vector1 'opposite' to vector2
    the threshold
    :param vector1:
    :param vector2:
    :param threshold:
    :return: if opposite
    """
    # L2 Normalization
    vector1 = vector1/np.linalg.norm(vector1)
    vector2 = vector2/np.linalg.norm(vector2)
    return True if vector1.dot(vector2.T) <= np.cos(np.deg2rad(threshold)) else False

def clamp(value, low, high):
    """
    clip(low,high)
    """
    return max(min(value, high), low)

if __name__ == '__main__':
    # FOR TEST
    pass
