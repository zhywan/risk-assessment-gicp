import numpy as np
import matplotlib.pyplot as plt
import time

# configure
age_group_count = np.genfromtxt('data/age_group_count.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
age_group_dist = age_group_count / sum(age_group_count)
race_sex_age = np.genfromtxt('data/race_sex_age_hipaa.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
race_sex_ratio = np.sum(race_sex_age, axis=0) / np.sum(race_sex_age)
race_ratio = race_sex_ratio[0::2] + race_sex_ratio[1::2]
race_ethnicity = np.genfromtxt('data/race_ethnicity.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
n_race = int(np.shape(race_sex_age)[1] / 2)
n_age = np.shape(race_sex_age)[0]
n_simulation = 10
size_begin = 10000
size_end = 210000
size_step = 10000
n_size = int((size_end - size_begin) / size_step)

Result_folder = "Results/"
print(Result_folder)

start1 = time.time()
result = np.zeros(n_size)
result_std = np.zeros(n_size)
for i_size in range(n_size):
    n_patient = size_begin + size_step * i_size
    print("Size of dataset: " + str(n_patient))
    # For each size of dataset
    risk = np.zeros(n_simulation)
    for i_simulation in range(n_simulation):
        # For each simulation
        print("Simulation #" + str(i_simulation))
        np.random.seed(i_size * n_simulation + i_simulation)  # set a new random seed
        patient_array = np.zeros((n_patient, 4)).astype(int)  # creat a empty array
        # generate the counts for each race
        patient_array[:, 0] = np.sort(np.random.choice(n_race, n_patient, p=race_ratio))  # set the race column
        base_index_race = 0  # counter for the number of patients assigned race values
        base_ethnicity = 0  # counter for the number of ethnicity values used
        for i_race in range(n_race):
            # For each race
            n_patient_race = np.count_nonzero(patient_array[:, 0] == i_race)  # number of patients with this race
            if n_patient_race == 0:
                continue
            sex_ratio_pre = race_sex_ratio[i_race*2:(i_race*2+2)]
            sex_ratio = sex_ratio_pre / np.sum(sex_ratio_pre)
            patient_array[base_index_race:(base_index_race + n_patient_race), 1] = np.sort(
                np.random.choice(2, n_patient_race, p=sex_ratio))  # set the sex column
            base_index_race_sex = base_index_race  # counter for number of patients assigned race and sex values
            for i_sex in range(2):
                # For each sex
                age_count = race_sex_age[:, (i_race*2 + i_sex)]
                age_ratio_pre = age_count / sum(age_count)
                age_ratio = np.zeros(n_age)
                for i in range(14):
                    sum_age_ratio_pre = 0
                    sum_age_group_dist = 0
                    if i == 13:
                        t = 25
                    else:
                        t = 5
                    for j in range(t):
                        k = i * 5 + j
                        sum_age_ratio_pre += age_ratio_pre[k]
                        sum_age_group_dist += age_group_dist[k]
                    for j in range(t):
                        k = i * 5 + j
                        age_ratio[k] = age_ratio_pre[k] / sum_age_ratio_pre * sum_age_group_dist
                age_ratio[n_age - 1] = 1 - np.sum(age_ratio)
                # calculate the number of patients with this race and sex
                n_patient_race_sex = np.count_nonzero(
                    patient_array[base_index_race:(base_index_race + n_patient_race), 1] == i_sex)
                patient_array[base_index_race_sex:(base_index_race_sex + n_patient_race_sex), 2] = np.sort(
                    np.random.choice(n_age, n_patient_race_sex, p=age_ratio))  # set the age column
                base_index_race_sex += n_patient_race_sex

            n_ethnicity_race = np.count_nonzero(race_ethnicity[:, 0] == i_race)  # number of ethnicity with this race
            ethnicity_count = race_ethnicity[base_ethnicity:(base_ethnicity + n_ethnicity_race), 1]
            ethnicity_ratio = ethnicity_count / sum(ethnicity_count)
            patient_array[base_index_race:(base_index_race + n_patient_race), 3] = np.random.choice(
                np.arange(base_ethnicity, (base_ethnicity + n_ethnicity_race)), n_patient_race,
                p=ethnicity_ratio)  # set the ethnicity column

            base_ethnicity += n_ethnicity_race
            base_index_race += n_patient_race

        unique_rows = np.unique(patient_array, axis=0)
        n_group = np.shape(unique_rows)[0]
        group_size = np.zeros(n_group)
        for i in range(n_group):
            distance = np.sum(np.absolute(patient_array - unique_rows[i, :]), axis=1)
            group_size[i] = np.count_nonzero(distance == 0)
        n_bad_group = np.count_nonzero(group_size < 11)
        risk[i_simulation] = n_bad_group / n_patient
    result[i_size] = np.mean(risk)
    result_std[i_size] = np.std(risk)
elapsed1 = (time.time() - start1)
print("Time used: " + str(elapsed1) + " seconds.\n")
#plt.plot(list(range(size_begin, size_end, size_step)), result)
x_values = list(range(size_begin, size_end, size_step))

with open(Result_folder + "result_R5.txt", 'w') as f:
    f.write("Time used: " + str(elapsed1) + " seconds.\n")
    for i in range(n_size):
        f.write(str(x_values[i]) + "," + str(result[i]) + "," + str(result_std[i]) + "\n")

plt.errorbar(x_values, result, result_std, fmt='-o')
for i in range(1, n_size, 2):
    plt.text(x_values[i], result[i] + 0.01, "%0.3f" %result[i], ha="center")
plt.xlabel('Number of patients')
plt.ylabel('R5 risk')
plt.grid()
plt.show()
