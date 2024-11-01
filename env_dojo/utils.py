import numpy
import numpy as np

'''
utils for math tools used in creating env
find detailed description before every function
'''

# rotate a 2-d vector by an angle (in ANGULAR)
def rotate_vector(vector: np.array([float, float]), angle: int) -> np.array:
    ang2rad = np.deg2rad(angle)
    r_mat = numpy.array([[np.cos(ang2rad), np.sin(ang2rad)],
                         [-np.sin(ang2rad), np.cos(ang2rad)]])
    post_vector = (r_mat.dot(vector)).astype('float')

    return post_vector


# return the included angle between angle_base and angle
# both in ANGULAR, and result in (-180, 180]
def angle_between(angle_base: int, angle: int) -> int:
    i_angle = angle_base - angle
    i_angle = (i_angle + 180) % 360 - 180

    return i_angle


# return the Euclidean distance between two point
def distance_between(dot1: np.array, dot2: np.array) -> float:
    dis = np.sqrt(np.sum(np.square(dot1 - dot2))).astype('float')
    return dis


# return the cross product of two 2-d vector
# the result determines whether vector2 rotates clockwise or counterclockwise to vector1
def cross_product(vector1: np.array([float, float]), vector2: np.array([float, float])) -> float:
    output = vector1[0] * vector2[1] - vector2[0] * vector1[1]
    return output


# return if vector1 cross vector2
# the vector should be expressed in absolute coordinate
def if_cross(vector1: np.array([[float, float], [float, float]]),
             vector2: np.array([[float, float], [float, float]])) -> bool:
    vector_1s_2s = vector1[0] - vector2[0]
    vector_1s_2e = vector1[0] - vector2[1]
    vector_1s_1e = vector1[0] - vector1[1]
    output = cross_product(vector_1s_2s, vector_1s_1e) * cross_product(vector_1s_2e, vector_1s_1e)
    cross = True if output < 0 else False
    return cross


# return if two vector straddle each other
# the vector should be expressed in absolute coordinate
def if_straddle(vector1: np.array([[float, float], [float, float]]),
                vector2: np.array([[float, float], [float, float]])) -> bool:
    straddle = if_cross(vector1, vector2) and if_cross(vector2, vector1)
    return straddle


# return if the included angle between vec1 and vec2 above threshold
def is_vector_opposite(vector1: np.array([float, float]),
                       vector2: np.array([float, float]),
                       threshold: int = 170) -> bool:
    # L2 Normalization
    v1_norm = np.linalg.norm(vector1)
    vector1 = vector1/v1_norm
    v2_norm = np.linalg.norm(vector2)
    vector2 = vector2/v2_norm

    outcome = vector1.dot(vector2.T)

    is_opposite = True if outcome <= np.cos(np.deg2rad(threshold)) else False

    return is_opposite


if __name__ == '__main__':
    # FOR TEST
    pass
