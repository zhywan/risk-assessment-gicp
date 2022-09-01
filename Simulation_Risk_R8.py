import numpy as np
import matplotlib.pyplot as plt
import time

# configure
file_suffix = "R8"
aaf = np.genfromtxt('data/AAF5.csv', delimiter=',', dtype=float, encoding='utf-8-sig')
max_n_snp = aaf.size
n_simulation = 100
size_begin = 100
size_end = 2100
size_step = 100
snp_begin = 1000
snp_end = 11000
snp_step = 2000

delta = 1e-06  # sequencing error rate
beta = 0.05
rate_control = 10
#mini = 1e-308

n_size = np.ceil((size_end - size_begin) / size_step).astype(int)
m_n_snp = np.ceil((snp_end - snp_begin) / snp_step).astype(int)

start1 = time.time()
result = np.zeros((n_size, m_n_snp))
result_std = np.zeros((n_size, m_n_snp))
for i_size in range(n_size):
    n_patient = size_begin + size_step * i_size
    print("Size of dataset: " + str(n_patient))
    n_control = n_patient * rate_control
    # For each size of dataset
    risk = np.zeros((n_simulation, m_n_snp))
    for i_simulation in range(n_simulation):
        # For each simulation
        print("Simulation #" + str(i_simulation))
        np.random.seed(i_size * n_simulation + i_simulation)  # set a new random seed

        # creat the patient and control datasets
        patient_array = np.zeros((n_patient, max_n_snp)).astype(bool)  # creat a empty array
        control_array = np.zeros((n_control, max_n_snp)).astype(bool)

        # creat the error for patient and control datasets
        patient_error = np.zeros((n_patient, max_n_snp)).astype(bool)  # creat a empty array
        control_error = np.zeros((n_control, max_n_snp)).astype(bool)

        # generate the data
        for i_snp in range(max_n_snp):
            patient_array[:, i_snp] = np.logical_or(
                np.random.choice([True, False], n_patient, p=[aaf[i_snp], 1 - aaf[i_snp]]),
                np.random.choice([True, False], n_patient, p=[aaf[i_snp], 1 - aaf[i_snp]]))
            control_array[:, i_snp] = np.logical_or(
                np.random.choice([True, False], n_control, p=[aaf[i_snp], 1 - aaf[i_snp]]),
                np.random.choice([True, False], n_control, p=[aaf[i_snp], 1 - aaf[i_snp]]))

        # generate the error for the data sharer
        patient_error_sharer_pre = np.random.choice([True, False], n_patient * max_n_snp, p=[delta, 1 - delta])
        patient_error_sharer = np.reshape(patient_error_sharer_pre, (-1, max_n_snp))
        patient_error_sharer_pre = np.random.choice([True, False], n_patient * max_n_snp, p=[delta, 1 - delta])
        patient_error_sharer = np.logical_or(patient_error_sharer,
                                             np.reshape(patient_error_sharer_pre, (-1, max_n_snp)))
        patient_error_attacker_pre = np.random.choice([True, False], n_patient * max_n_snp, p=[delta, 1 - delta])
        patient_error_attacker = np.reshape(patient_error_attacker_pre, (-1, max_n_snp))
        patient_error_attacker_pre = np.random.choice([True, False], n_patient * max_n_snp, p=[delta, 1 - delta])
        patient_error_attacker = np.logical_or(patient_error_attacker,
                                               np.reshape(patient_error_attacker_pre, (-1, max_n_snp)))
        control_error_pre = np.random.choice([True, False], n_control * max_n_snp, p=[delta, 1 - delta])
        control_error = np.reshape(control_error_pre, (-1, max_n_snp))
        control_error_pre = np.random.choice([True, False], n_control * max_n_snp, p=[delta, 1 - delta])
        control_error = np.logical_or(control_error, np.reshape(control_error_pre, (-1, max_n_snp)))

        # generate data with error
        patient_array_sharer = np.logical_xor(patient_array, patient_error_sharer)
        beacon_x = np.sum(patient_array_sharer, axis=0) > 0
        patient_array_attacker = np.logical_xor(patient_array, patient_error_attacker)
        control_array_attacker = np.logical_xor(control_array, control_error)

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
            r3 = np.sum(patient_lrt < control_fp_theta) / n_patient
            risk[i_simulation, i_n_snp] = r3
    result[i_size, :] = np.mean(risk, axis=0)
    result_std[i_size, :] = np.std(risk, axis=0)

elapsed1 = (time.time() - start1)
print("Time used: " + str(elapsed1) + " seconds (loading).\n")

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
    plt.errorbar(x_values, result[:, i_n_snp], result_std[:, i_n_snp], fmt='-o', label=n_snp_label[i_n_snp])
plt.xlabel('Number of patients')
plt.ylabel('PK3 risk')
plt.legend(loc='upper right')
plt.grid()
# save figure
plt.savefig('Results/figure_' + file_suffix + '.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
