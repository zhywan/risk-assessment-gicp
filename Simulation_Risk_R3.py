import numpy as np
import matplotlib.pyplot as plt
import time

# configure
age_group_count = np.genfromtxt('data/age_group_count.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
age_group_dist = age_group_count / sum(age_group_count)
file_suffix = "R3"
race_sex_age = np.genfromtxt('data/race_sex_age_hipaa.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
race_sex_ratio = np.sum(race_sex_age, axis=0) / np.sum(race_sex_age)
race_ratio = race_sex_ratio[0::2] + race_sex_ratio[1::2]

race_sex_count = np.sum(race_sex_age, axis=0)
race_count = race_sex_count[0::2] + race_sex_count[1::2]

race_ethnicity = np.genfromtxt('data/race_ethnicity.csv', delimiter=',', dtype=int, encoding='utf-8-sig')
n_race = int(np.shape(race_sex_age)[1] / 2)
n_age = np.shape(race_sex_age)[0]
n_simulation = 100
size_begin = 10000
size_end = 410000
size_step = 10000
n_size = int((size_end - size_begin) / size_step)

Result_folder = "Results/"
print(Result_folder)

start1 = time.time()
result = np.zeros(n_size)
result_std = np.zeros(n_size)
result_race = np.zeros((n_race, n_size))
result_std_race = np.zeros((n_race, n_size))
for i_size in range(n_size):
    n_patient = size_begin + size_step * i_size
    print("Size of dataset: " + str(n_patient))
    # For each size of dataset
    risk = np.zeros(n_simulation)
    risk_race = np.zeros((n_race, n_simulation))
    for i_simulation in range(n_simulation):
        # For each simulation
        print("Simulation #" + str(i_simulation))
        np.random.seed(i_size * n_simulation + i_simulation)  # set a new random seed

        # creat the patient dataset
        patient_array = np.zeros((n_patient, 4)).astype(int)  # creat a empty array
        # generate the counts for each race
        patient_array[:, 0] = np.sort(np.random.choice(n_race, n_patient, p=race_ratio))  # set the race column
        base_index_race = 0  # counter for the number of patients assigned race values
        base_ethnicity = 0  # counter for the number of ethnicity values used
        for i_race in range(n_race):
            # For each race
            n_patient_race = np.count_nonzero(patient_array[:, 0] == i_race)  # number of patients with this race
            n_ethnicity_race = np.count_nonzero(race_ethnicity[:, 0] == i_race)  # number of ethnicity with this race
            if n_patient_race == 0:
                base_ethnicity += n_ethnicity_race
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

            ethnicity_count = race_ethnicity[base_ethnicity:(base_ethnicity + n_ethnicity_race), 1]
            ethnicity_ratio = ethnicity_count / sum(ethnicity_count)
            patient_array[base_index_race:(base_index_race + n_patient_race), 3] = np.random.choice(
                np.arange(base_ethnicity, (base_ethnicity + n_ethnicity_race)), n_patient_race,
                p=ethnicity_ratio)  # set the ethnicity column

            base_ethnicity += n_ethnicity_race
            base_index_race += n_patient_race

        # creat the lookup dictionary
        dic_group_size = {}
        base_ethnicity = 0  # counter for the number of ethnicity values used
        base_index_race = 0  # counter for the number of patients assigned race values
        for i_race in range(n_race):
            n_ethnicity_race = np.count_nonzero(race_ethnicity[:, 0] == i_race)  # number of ethnicity with this race
            for i_sex in range(2):
                for i_age in range(n_age):
                    pop_race_sex_age = race_sex_age[i_age, i_race * 2 + i_sex]
                    for i_ethnicity in range(n_ethnicity_race):
                        tuple_demo = (i_race, i_sex, i_age, base_ethnicity + i_ethnicity)
                        pop_ethnicity = race_ethnicity[base_ethnicity + i_ethnicity, 1]
                        pop_race_sex_age_ethnicity = np.ceil(pop_race_sex_age / race_count[i_race] * pop_ethnicity).astype(int)
                        dic_group_size[tuple_demo] = pop_race_sex_age_ethnicity
            base_ethnicity += n_ethnicity_race

            # lookup from the dictionary
            n_patient_race = np.count_nonzero(patient_array[:, 0] == i_race)  # number of patients with this race
            journalist_risk_race = 0
            for i_patient in range(base_index_race, (base_index_race + n_patient_race)):
                tuple_patient = tuple(patient_array[i_patient, :])
                journalist_risk_race_i = 1 / dic_group_size[tuple_patient]
                if journalist_risk_race_i > journalist_risk_race:
                    journalist_risk_race = journalist_risk_race_i
                    if journalist_risk_race_i == 1:
                        break
            risk_race[i_race, i_simulation] = journalist_risk_race
            base_index_race += n_patient_race


        # lookup from the dictionary
        journalist_risk = 0
        for i_patient in range(n_patient):
            tuple_patient = tuple(patient_array[i_patient, :])
            journalist_risk_i = 1/dic_group_size[tuple_patient]
            if journalist_risk_i > journalist_risk:
                journalist_risk = journalist_risk_i
                if journalist_risk_i == 1:
                    break
        #print("Journalist risk: " + str(journalist_risk))
        risk[i_simulation] = journalist_risk
    result[i_size] = np.mean(risk)
    result_std[i_size] = np.std(risk)
    for i_race in range(n_race):
        result_race[i_race, i_size] = np.mean(risk_race[i_race, :])
        result_std_race[i_race, i_size] = np.std(risk_race[i_race, :])

elapsed1 = (time.time() - start1)
print("Time used: " + str(elapsed1) + " seconds (loading).\n")

# save results
with open(Result_folder + 'risks_' + file_suffix + '.npy', 'wb') as f:
    np.save(f, risk)
with open(Result_folder + 'results_' + file_suffix + '.npy', 'wb') as f:
    np.save(f, result)
    np.save(f, result_std)

with open(Result_folder + 'risks_' + file_suffix + '_race.npy', 'wb') as f:
    np.save(f, risk_race)
with open(Result_folder + 'results_' + file_suffix + '_race.npy', 'wb') as f:
    np.save(f, result_race)
    np.save(f, result_std_race)

# plot figure 1
plt.figure()
x_values = list(range(size_begin, size_end, size_step))

with open(Result_folder + "result_" + file_suffix + ".txt", 'w') as f:
    f.write("Time used: " + str(elapsed1) + " seconds.\n")
    for i in range(n_size):
        f.write(str(x_values[i]) + "," + str(result[i]) + "," + str(result_std[i]) + "\n")

plt.errorbar(x_values, result, result_std, fmt='-o')

plt.xlabel('Number of patients')
plt.ylabel(file_suffix + ' risk')
plt.grid()
# save figure
plt.savefig(Result_folder + 'figure_' + file_suffix + '.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

# plot figure 2
race_label = ['White', 'Black', 'American Indian and Alaska Native', 'Asian',
              'Native Hawaiian and Other Pacific Islander', 'Others']

with open(Result_folder + "result_" + file_suffix + "_race.txt", 'w') as f:
    for i_race in range(n_race):
        f.write(race_label[i_race] + "\n")
        for i in range(n_size):
            f.write(str(x_values[i]) + "," + str(result_race[i_race, i]) + "," + str(result_std_race[i_race, i]) + "\n")

plt.figure()
for i_race in range(n_race):
    x_values = list(range(size_begin, size_end, size_step))
    plt.errorbar(x_values, result_race[i_race, :], result_std_race[i_race, :], fmt='-o', label=race_label[i_race])
plt.xlabel('Number of patients')
plt.ylabel(file_suffix + ' risk')
plt.legend(loc='upper right')
plt.grid()
# save figure
plt.savefig(Result_folder + 'figure_' + file_suffix + '_race.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
