import numpy as np
import matplotlib.pyplot as plt
import time

# configure
file_suffix = "R9"
aaf = np.genfromtxt('data/AAF5.csv', delimiter=',', dtype=float, encoding='utf-8-sig')
max_n_snp = aaf.size
n_simulation = 10
size_begin = 500
size_end = 30500
size_step = 500
snp_begin = 100
snp_end = 2000
snp_step = 400

delta = 1e-06  # sequencing error rate
beta = 0.05
rate_control = 10
n_phe = 10
#mini = 1e-308

n_size = np.ceil((size_end - size_begin) / size_step).astype(int)
m_n_snp = np.ceil((snp_end - snp_begin) / snp_step).astype(int)

start1 = time.time()
result = np.zeros((n_size, m_n_snp))
result_std = np.zeros((n_size, m_n_snp))
for i_size in range(n_size):
    n_individual = size_begin + size_step * i_size
    print("Size of dataset: " + str(n_individual))
    # For each size of dataset
    risk = np.zeros((n_simulation, m_n_snp))
    for i_simulation in range(n_simulation):
        # For each simulation
        print("Simulation #" + str(i_simulation))
        np.random.seed(i_size * n_simulation + i_simulation)  # set a new random seed
        # record corrected inference
        correct_array = np.ones((n_individual, m_n_snp)).astype(bool)  # creat a empty array
        # creat the patient and control datasets
        individual_array = np.zeros((n_individual, max_n_snp)).astype(bool)  # creat a empty array

        # creat the error for patient and control datasets
        individual_error = np.zeros((n_individual, max_n_snp)).astype(bool)  # creat a empty array

        # generate the data
        for i_snp in range(max_n_snp):
            individual_array[:, i_snp] = np.logical_or(
                np.random.choice([True, False], n_individual, p=[aaf[i_snp], 1 - aaf[i_snp]]),
                np.random.choice([True, False], n_individual, p=[aaf[i_snp], 1 - aaf[i_snp]]))

        # generate the error for the data sharer
        individual_error_sharer_pre = np.random.choice([True, False], n_individual * max_n_snp, p=[delta, 1 - delta])
        individual_error_sharer = np.reshape(individual_error_sharer_pre, (-1, max_n_snp))
        individual_error_sharer_pre = np.random.choice([True, False], n_individual * max_n_snp, p=[delta, 1 - delta])
        individual_error_sharer = np.logical_or(individual_error_sharer,
                                             np.reshape(individual_error_sharer_pre, (-1, max_n_snp)))
        individual_error_attacker_pre = np.random.choice([True, False], n_individual * max_n_snp, p=[delta, 1 - delta])
        individual_error_attacker = np.reshape(individual_error_attacker_pre, (-1, max_n_snp))
        individual_error_attacker_pre = np.random.choice([True, False], n_individual * max_n_snp, p=[delta, 1 - delta])
        individual_error_attacker = np.logical_or(individual_error_attacker,
                                               np.reshape(individual_error_attacker_pre, (-1, max_n_snp)))

        # generate data with error
        individual_array_sharer = np.logical_xor(individual_array, individual_error_sharer)
        individual_array_attacker = np.logical_xor(individual_array, individual_error_attacker)
        for i_phe in range(n_phe):
            n_patient = int((i_phe + 1) * n_individual / 100)
            n_control = n_individual - n_patient
            index_individual = np.random.permutation(n_individual)
            index_patient = index_individual[0:n_patient]
            index_control = index_individual[n_patient:n_individual]
            patient_array_sharer = individual_array_sharer[index_patient, :]
            patient_array_attacker = individual_array_attacker[index_patient, :]
            control_array_attacker = individual_array_attacker[index_control, :]
            beacon_x = np.sum(patient_array_sharer, axis=0) > 0
            # calculate lrt
            #D0 = np.power(1 - aaf, 2 * n_patient)
            D0 = 1 - aaf
            for i in range(2 * n_patient - 1):
                D0 *= (1 - aaf)
            #D1 = delta * np.power(1 - aaf, 2 * n_patient - 2)
            D1 = delta * (1 - aaf)
            for i in range(2 * n_patient - 3):
                D1 *= (1 - aaf)

            D2 = 1 / delta * np.power(1 - aaf, 2)
            #log_1 = np.log10(np.divide(1 - D0, 1 - D1))
            log_1 = np.log10(1 - D0) - np.log10(1 - D1)
            log_0 = np.log10(D2)
            lamb = beacon_x * log_1 + (1 - beacon_x) * log_0
            lamb_mat_patient = np.dot(np.ones((n_patient, 1)), np.reshape(lamb, (1, -1)))
            patient_lrt_pre = patient_array_attacker * lamb_mat_patient
            lamb_mat_control = np.dot(np.ones((n_control, 1)), np.reshape(lamb, (1, -1)))
            control_lrt_pre = control_array_attacker * lamb_mat_control

            for i_n_snp in range(m_n_snp):
                # For each number of SNPs
                n_snp = snp_begin + snp_step * i_n_snp
                patient_lrt = np.sum(patient_lrt_pre[:, 1:n_snp], axis=1)
                control_lrt = np.sum(control_lrt_pre[:, 1:n_snp], axis=1)
                sorted_control_lrt = np.sort(control_lrt)
                control_fp_theta = sorted_control_lrt[np.ceil(n_control * beta).astype(int)]

                correct_patient_lrt = patient_lrt < control_fp_theta
                correct_array[index_patient[correct_patient_lrt], i_n_snp] = False
                correct_control_lrt = control_lrt >= control_fp_theta
                correct_array[index_control[correct_control_lrt], i_n_snp] = False
        for i_n_snp in range(m_n_snp):
            # For each number of SNPs
            r3 = np.sum(correct_array[:, i_n_snp]) / n_individual
            risk[i_simulation, i_n_snp] = r3
    result[i_size, :] = np.mean(risk, axis=0)
    result_std[i_size, :] = np.std(risk, axis=0)

elapsed1 = (time.time() - start1)
print("Time used: " + str(elapsed1) + " seconds.\n")

# save results
with open('Results/risks_' + file_suffix + '.npy', 'wb') as f:
    np.save(f, risk)
with open('Results/results_' + file_suffix + '.npy', 'wb') as f:
    np.save(f, result)
    np.save(f, result_std)

# plot figure
n_snp_label = []
for i_n_snp in range(m_n_snp):
    n_snp_label.append(str(snp_begin + snp_step * i_n_snp) + ' SNPs')
plt.figure()
x_values = list(range(size_begin, size_end, size_step))
for i_n_snp in range(m_n_snp):
    #plt.errorbar(x_values, result[:, i_n_snp], result_std[:, i_n_snp], fmt='-o', label=n_snp_label[i_n_snp])
    plt.plot(x_values, result[:, i_n_snp], '-o', label=n_snp_label[i_n_snp])
plt.xlabel('Number of patients')
plt.ylabel('R9 risk')
plt.legend(loc='upper left')
plt.grid()
# save figure
plt.savefig('Results/figure_' + file_suffix + '_lineplot.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
