# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map

# def target_mean_v2(data, y_name, x_name):
#     result = np.zeros(data.shape[0])
#     value_dict = dict()
#     count_dict = dict()
#     for i in range(data.shape[0]):
#         if data.loc[i, x_name] not in value_dict.keys():
#             value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
#             count_dict[data.loc[i, x_name]] = 1
#         else:
#             value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
#             count_dict[data.loc[i, x_name]] += 1
#     for i in range(data.shape[0]):
#         result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
#     return result


cpdef double target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result


cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef unordered_map[double, double] value
    cdef unordered_map[double, double] count

    cdef unordered_map[double, double].iterator vend = value.end()
    cdef long i
    for i in range(nrow):
        if value.find(x[i]) != vend:
            value[x[i]] = y[i]
            count[x[i]] = 1
        else:
            value[x[i]] += y[i]
            count[x[i]] += 1

    i = 0
    for i in range(nrow):
        result[i] = (value[x[i]] - y[i])/(count[x[i]] - 1)

