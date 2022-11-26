import matplotlib.pyplot as plt
import numpy as np

def read_act_data(file_name):
    data_dict, name = dict(), ""
    file = open(file_name, 'r')
    for line in file.readlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.replace('.','',1).isnumeric():
            data_dict[name].append(float(line))
        else:
            name = line
            data_dict[name] = list()
    return data_dict


def plot(file_name, out_put):
    plt.style.use('seaborn-paper')
    plt.figure(1, facecolor="white")
    plt.cla()
    x = np.arange(0, 1500, 5 )
    data_dict = read_act_data(file_name)
    for name in data_dict.keys():
        plt.plot(x, data_dict[name], label=name)

    plt.ylabel("ROC-AUC (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_put, dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    plot("act_data.txt", './gipa_act.png')
