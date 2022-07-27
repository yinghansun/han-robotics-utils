from typing import Optional, Union
import numpy as np


def vector_format_standardization(
    vector: Union[list, np.ndarray], std_type: Optional[str] = 'list'
) -> Union[list, np.ndarray]:
    ''' A function for standardize the format of vectors.

    Args:
    - vec: The vector you want to convert.
    - std_type: The type you want to convert the vector, could be 
        either 'list' or '1darray'.

    Returns:
    The standardized vector.
    '''

    assert type(vector) == list or type(vector) == np.ndarray, \
        'the parameter \'vector\' must be either a list or a numpy array.'
    assert std_type == 'list' or std_type == '1darray', \
        'the parameter \'std_type\' must be either \'list\' or \'1darray\'.'

    if std_type == 'list':
        if type(vector) == np.ndarray:
            vec_shape = vector.shape
            assert len(vec_shape) == 1 or vec_shape[0] == 1 or vec_shape[1] == 1

            if len(vec_shape) == 1:
                vector = list(vector)
            elif vec_shape[0] == 1:
                vector = list(vector[0])
            elif vec_shape[1] == 1:
                num_elements = vec_shape[0]
                vector = vector.reshape(1, num_elements)
                vector = list(vector[0])

        return vector

    else:
        if type(vector) == list:
            vector = np.array(vector)
        else:
            vec_shape = vector.shape
            assert len(vec_shape) == 1 or vec_shape[0] == 1 or vec_shape[1] == 1

            if len(vec_shape) != 1:
                vector = vector.flatten()

        return vector


def _test_standardization_func():
    # wrong_vector = 3
    # wrong_std_type = 'wrong_type'
    # vector_format_standardization(wrong_vector)
    # vector_format_standardization([1, 2], wrong_std_type)

    print('----------')
    print('test list to 1d numpy array')
    a_list = [1, 2, 3]
    a_1darray = vector_format_standardization(a_list, std_type='1darray')
    print(type(a_1darray), a_1darray)

    print('----------')
    print('test 2d numpy array to 1d numpy array')
    b_2darray = np.array([[1, 2, 3]])
    print(b_2darray.shape, b_2darray)
    b_1darray = vector_format_standardization(b_2darray, std_type='1darray')
    print(type(b_1darray), b_1darray.shape, b_1darray)
    
    c_2darray = np.array([[4], [5], [6]])
    print(c_2darray.shape, c_2darray)
    c_1darray = vector_format_standardization(c_2darray, std_type='1darray')
    print(type(c_1darray), c_1darray.shape, c_1darray)

    print('----------')
    print('test 1d numpy array to list')
    d_1darray = np.array([1, 2, 3])
    d_list = vector_format_standardization(d_1darray)
    print(type(d_list), d_list)

    print('----------')
    print('test 2d numpy array to list')
    e_2darray = np.array([[1, 2, 3]])
    e_list = vector_format_standardization(e_2darray)
    print(type(e_list), e_list)
    
    f_2darray = np.array([[4], [5], [6]])
    f_list = vector_format_standardization(f_2darray)
    print(type(f_list), f_list)

    print('----------')
    print('test list to list')
    g_list = [1, 2, 3]
    g_list = vector_format_standardization(g_list)
    print(type(g_list), g_list)

    print('----------')
    print('test 1darray to 1darray')
    h_1darray = np.array([1, 2, 3])
    h_1darray = vector_format_standardization(h_1darray, std_type='1darray')
    print(type(h_1darray), h_1darray)


if __name__ == '__main__':
    _test_standardization_func()