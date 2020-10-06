import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

class Predictor():
    def __init__(self, criteria,csv_x_path, csv_y_path, real_path_x, real_path_y):
        self.criteria = criteria
        self.my_csv_x = pd.read_csv(csv_x_path)
        self.data_x = self.my_csv_x.values
        self.my_csv_y = pd.read_csv(csv_y_path)
        self.data_y = self.my_csv_y.values
        self.real_path_x = real_path_x
        self.real_path_y = real_path_y
        self.dataset_size = len(self.data_x)
        self.online_path_x = real_path_x.copy()
        self.online_path_y = real_path_y.copy()
        self.si = 0.0
        self.online_step = 90
        self.slop_step = 10
        self.cost_function_coef = 1
        self.captue_size = 100

        for cnt in range(self.dataset_size):
            self.data_x[cnt][1] = float(self.data_x[cnt][1])
            self.data_y[cnt][1] = float(self.data_y[cnt][1])

    def offline_predictor(self):
        diff_array_x = self.data_x
        diff_array_y = self.data_y
        diff_array = np.zeros([self.dataset_size, self.captue_size])
        a_array_x = []
        a_array_y = []
        b_array_x = []
        b_array_y = []
        a_temp = []
        b_temp = []

        for cnt_1 in range(self.dataset_size):
            is_accurate = True
            for cnt_2 in range(1, self.criteria):
                x_diff = self.real_path_x[cnt_2] - self.data_x[cnt_1][cnt_2]
                y_diff = self.real_path_y[cnt_2] - self.data_y[cnt_1][cnt_2]
                diff_array_x[cnt_1][cnt_2] = x_diff
                diff_array_y[cnt_1][cnt_2] = y_diff
                diff_array[cnt_1][cnt_2] = math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))
                # sign = self.sign_calc(diff_array[cnt_1][cnt_2])
                sign = self.sign_calc(y_diff)
                if(cnt_2 == 1):
                    selected_sign = sign
                else :
                    if(selected_sign != sign):
                        is_accurate = False
                        break
            if(is_accurate == True):
                # 2 for B below the line, 1 for A above the line
                if(selected_sign == 1):
                    diff_array[cnt_1][0] = 2
                else :
                    diff_array[cnt_1][0] = 1
            else :
                diff_array[cnt_1][0] = 0

        for cnt_3 in range(self.dataset_size):
            if diff_array[cnt_3][0] == 2 : #Checks if it is in B
                b_array_x.append(diff_array_x[cnt_3])
                b_array_y.append(diff_array_y[cnt_3])
                b_temp.append(diff_array[cnt_3])

            elif diff_array[cnt_3][0] == 1 :
                a_array_x.append(diff_array_x[cnt_3])
                a_array_y.append(diff_array_y[cnt_3])
                a_temp.append(diff_array[cnt_3])

        # a_array_selected_index = self.min_index(a_array, 'a')
        # b_array_selected_index = self.min_index(b_array, 'b')

        # a_array_selected_index = self.min_index(a_temp, 'a')
        # b_array_selected_index = self.min_index(b_temp, 'b')

        a_array_selected_index = self.diff_slope(a_temp, a_array_x, a_array_y, 'a', self.slop_step)
        b_array_selected_index = self.diff_slope(b_temp, b_array_x, b_array_y, 'b', self.slop_step)

        self.wa = self.db / (self.da + self.db)
        self.wb = self.da / (self.da + self.db)

        result_array_x = []
        result_array_y = []

        for res_cnt in range(self.criteria, len(a_array_x[a_array_selected_index])):
            # result_array.append(self.wa * a_array[a_array_selected_index][res_cnt] + self.wb * b_array[b_array_selected_index][res_cnt])
            result_array_x.append(self.wa * a_array_x[a_array_selected_index][res_cnt] + self.wb * b_array_x[b_array_selected_index][res_cnt])
            result_array_y.append(self.wa * a_array_y[a_array_selected_index][res_cnt] + self.wb * b_array_y[b_array_selected_index][res_cnt])

        return (result_array_x, result_array_y, a_array_x, a_array_y, a_array_selected_index, b_array_x, b_array_y, b_array_selected_index)

    def sign_calc(self, number):
        if number >= 0:
            return 1
        else:
            return 0

    def min_index(self, array, type):
        #arrays_len = len(array)
        arrays_sum = []
        for cnt_1 in range(len(array)):
            sum = 0
            for cnt_2 in range(1, self.criteria):
                sum += abs(array[cnt_1][cnt_2])
            arrays_sum.append(sum)
        if(type == 'a'):
            self.da = min(arrays_sum)
        elif(type == 'b'):
            self.db = min(arrays_sum)

        return arrays_sum.index(min(arrays_sum))

    def diff_slope(self, array, array_x, array_y, type, step):
        diff_array = []
        slop_array = []
        for cnt_1 in range(len(array)):
            sum = 0
            slop = (array_y[cnt_1][self.criteria - 1] - array_y[cnt_1][self.criteria - 1 - step]) / (array_x[cnt_1][self.criteria - 1] - array_x[cnt_1][self.criteria - 1 - step])
            real_path_slop = (self.real_path_y[self.criteria - 1] - self.real_path_y[self.criteria - 1 - step]) / (self.real_path_x[self.criteria - 1] - self.real_path_x[self.criteria - 1 - step])
            slop_array.append(real_path_slop - slop)
            #slop_array.append((array_y[self.criteria - 1][self.criteria - 1 - step]) / (array_x[self.criteria - 1][self.criteria - 1 - step]))
            for cnt_2 in range(1, self.criteria):
                sum += abs(array[cnt_1][cnt_2])
            diff_array.append(sum)
        selected_index = self.cost_function(diff_array, slop_array)
        if type == 'a' :
            self.da = diff_array[selected_index]
        elif type == 'b' :
            self.db = diff_array[selected_index]

        return selected_index


    def cost_function(self, diff_array, slop_array):
        #By increasing the coef, impact of slop increases
        coef = self.cost_function_coef
        results = []
        for cnt_1 in range(len(diff_array)):
            # sum = coef * math.pow(diff_array[cnt_1], 2) + math.pow(slop_array[cnt_1], 2)
            sum = coef * abs(diff_array[cnt_1] / max(diff_array)) + abs(slop_array[cnt_1] / max(slop_array))
            results.append(sum)
        return results.index(min(results))

    def online_predictor_x(self):
        online_predicted_results = []
        step = self.online_step
        for step_cnt in range(self.criteria, len(self.online_path_x) + 1, step):
            self.c = []
            self.r = [1.0]
            self.online_path_x = self.real_path_x.copy()
            print(step_cnt)
            self.c.append(self.expectation_x(0, step_cnt))
            self.c.append(self.expectation_x(1, step_cnt))

            self.r.append(self.c[1] / self.c[0])
            self.si = self.r[1] / self.r[0]

            for cnt in range(step_cnt, len(self.online_path_x)):
                self.online_path_x[cnt] = (2 + self.si) * self.online_path_x[cnt - 1] + (-1 + (-2) * self.si) * self.online_path_x[cnt - 2] + self.si * self.online_path_x[cnt - 3]
            online_predicted_results.append(self.online_path_x)
        # return self.online_path_x
        return online_predicted_results


    def expectation_x(self, k, in_step):
        sum = 0.0
        for cnt in range(k, in_step):
            sum += self.online_path_x[cnt] * self.online_path_x[cnt - k]
        sum /= in_step
        return sum

    def online_predictor_y(self):
        s = []
        for p in range(0, 100):
            s.append(p)

        online_predicted_results = []
        step = self.online_step
        for step_cnt in range(self.criteria, len(self.online_path_y) + 1, step):
            self.c = []
            self.r = [1.0]
            self.online_path_y = self.real_path_y.copy()
            # print(step_cnt)
            self.c.append(self.expectation_y(0, step_cnt))
            self.c.append(self.expectation_y(1, step_cnt))

            self.r.append(self.c[1] / self.c[0])
            self.si = self.r[1] / self.r[0]

            for cnt in range(step_cnt, len(self.online_path_y)):
                self.online_path_y[cnt] = (2 + self.si) * self.online_path_y[cnt - 1] + (-1 + (-2) * self.si) * self.online_path_y[cnt - 2] + self.si * self.online_path_y[cnt - 3]
            online_predicted_results.append(self.online_path_y)
            # plt.plot(s, self.online_path_y, 'r--', s, self.real_path_y, 'b--')
        # plt.show()
        # return self.online_path_y
        return online_predicted_results

    def expectation_y(self, k, in_step):
        sum = 0.0
        for cnt in range(k, in_step):
            sum += self.online_path_y[cnt] * self.online_path_y[cnt - k]
        sum /= in_step
        return sum

    def get_online_step(self):
        return self.online_step

    def get_capture_size(self):
        return self.captue_size

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main():
    order = 5
    fs = 30.0  # sample rate, Hz
    cutoff = 2  # desired cutoff frequency of the filter, Hz

    path_file = open("./records/sample-2.txt", 'r')
    offline_path = path_file.readline()
    offline_path = eval(offline_path)

    offline_path = butter_lowpass_filter(offline_path, cutoff, fs, order)

    # print(offline_path)

    offline_path_x = path_file.readline()
    offline_path_x = eval(offline_path_x)

    offline_path_x = butter_lowpass_filter(offline_path_x, cutoff, fs, order)
    criteria = 10
    predictor = Predictor(criteria, "records/real-x.csv", "records/real-y.csv", offline_path_x, offline_path)
    res = predictor.offline_predictor()
    r_x, r_y, a_x, a_y, a_i, b_x, b_y, b_i = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]

    offline_result_x = r_x
    offline_result_y = r_y
    online_results_x = predictor.online_predictor_x()
    online_results_y = predictor.online_predictor_y()

    for cnt in range(len(online_results_x)):
        online_results_x[cnt] = online_results_x[cnt][criteria:]
        online_results_y[cnt] = online_results_y[cnt][criteria:]

    # Coefficient for online and offline, coef is for offline
    coef = 0.9


    offline_result_x[:] = [x * coef for x in offline_result_x]

    for online_result_x in online_results_x:
        online_result_x[:] = [x * (1 - coef) for x in online_result_x]

    offline_result_y[:] = [x * coef for x in offline_result_y]

    for online_result_y in online_results_y:
        online_result_y[:] = [x * (1 - coef) for x in online_result_y]


    final_results_x = []
    final_results_y = []

    final_discrete_results_x = []
    final_discrete_results_y = []

    discrete_cnt = 0
    online_step = predictor.get_online_step()

    for online_result_x in online_results_x :
        final_result_x = [x + y for x,y in zip(offline_result_x, online_result_x)]
        final_results_x.append(final_result_x)
        base = discrete_cnt * online_step
        final_discrete_results_x.append(final_result_x[base:base + online_step])
        discrete_cnt += 1

    discrete_cnt = 0
    for online_result_y in online_results_y:
        final_result_y = [x + y for x,y in zip(offline_result_y, online_result_y)]
        final_results_y.append(final_result_y)
        base = discrete_cnt * online_step
        final_discrete_results_y.append(final_result_y[base:base + online_step])
        discrete_cnt += 1

    # for cnt in range(len(final_results_y)):
        # plt.plot(final_results_x[cnt], final_results_y[cnt], 'b--')
    # print(final_results_x[0])
    RMSE_numbers = []
    RMSE_sum = 0.0
    RMSE_result = 0.0
    plots = []
    plots.append(plt.plot([], [], 'r--', label="Predicted Trajectory")[0])
    for cnt in range(len(final_discrete_results_y)):
        plt.plot(final_discrete_results_x[cnt], final_discrete_results_y[cnt], 'r--')
        rmse_range = criteria + (cnt * online_step)
        # print(rmse_range)
        if(rmse_range != 100):
            for cnt_2 in range(rmse_range, rmse_range + online_step):
                # print(len(final_discrete_results_x[cnt]))
                # print(len(offline_path_x))
                rmse_diff_x = final_discrete_results_x[cnt][cnt_2 - rmse_range] - offline_path_x[cnt_2]
                rmse_diff_y = final_discrete_results_y[cnt][cnt_2 - rmse_range] - offline_path[cnt_2]
                rmse_diff = math.pow(rmse_diff_x, 2) + math.pow(rmse_diff_y, 2)
                RMSE_numbers.append(rmse_diff)
    for RMSE_number in RMSE_numbers :
        RMSE_sum += RMSE_number
    RMSE_result = math.sqrt(RMSE_sum / (predictor.get_capture_size() - criteria))
    MSE_result = RMSE_sum / (predictor.get_capture_size() - criteria)

    r_path_diameter_x = offline_path_x[predictor.get_capture_size() - 1] - offline_path_x[criteria]
    r_path_diameter_y = offline_path[predictor.get_capture_size() - 1] - offline_path[criteria]
    r_path_diameter = math.sqrt(math.pow(r_path_diameter_x, 2) + math.pow(r_path_diameter_y, 2))
    N_RMSE_result = RMSE_result / r_path_diameter

    N_MSE_result = MSE_result / (r_path_diameter_x * r_path_diameter_y)
    print("RMSE is : ", RMSE_result)
    # print(N_RMSE_result)
    # print(N_MSE_result)

    path_distance = 0.0
    for path_cnt in range(1, len(offline_path_x)) :
        offline_path_diff_x = offline_path_x[path_cnt] - offline_path_x[path_cnt - 1]
        offline_path_diff_y = offline_path[path_cnt] - offline_path[path_cnt - 1]
        path_distance += math.sqrt(math.pow(offline_path_diff_x, 2) + math.pow(offline_path_diff_y, 2))

    print("Total path is : ", path_distance)
    print("Error rmse/distance :", RMSE_result / path_distance)
    plots.append(plt.plot(offline_path_x[criteria:], offline_path[criteria:], 'g--', label = 'Actual Trajectory')[0])
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(color='black', linestyle='--', linewidth=0.4)
    plt.legend(handles=plots)
    # plt.plot(final_result_x[criteria:], final_result_y[criteria:], 'r--', offline_path_x[criteria:], offline_path[criteria:], 'b--')
    # plt.plot(final_results_x[criteria:], final_results_y[criteria:], 'r--', offline_path_x[criteria:], offline_path[criteria:], 'g--',
    #          a_x[a_i][criteria:], a_y[a_i][criteria:], 'b--', b_x[b_i][criteria:], b_y[b_i][criteria:], 'b--')
    # plt.plot(r_x, r_y, 'r--', a_x[a_i], a_y[a_i], 'b--', b_x[b_i], b_y[b_i], 'b--', offline_path_x, offline_path, 'g--')
    # plt.xticks([4,5,6,7,8,9,10,12])
    plt.show()
main()