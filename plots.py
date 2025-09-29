import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import numpy as np
from config import D, N, n_sim, lam, n_values
import argparse


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "text.usetex": False,

    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.frameon": False,
    "figure.titlesize": 16
})

def plot(dir_list, labels):
    folder_name = 'photos'
    os.makedirs(folder_name, exist_ok=True)

    gaussian_color  = plt.get_cmap('Set1').colors[0]
    student_labels = [lab for lab in labels if "Student" in lab]

    student_colors = {
    0: '#4C72B0',
    1: '#55A868',
    2: '#FF8C00'
    }
    color_map = {}

    for lab in labels:
        if "Student" not in lab:
            color_map[lab] = gaussian_color
        else:
            idx = student_labels.index(lab)
            color_map[lab] = student_colors[idx]
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), sharex=True, sharey=True)

    for dir, label in zip(dir_list, labels):
        value_diff_mean_list = []
        value_diff_std_list = []
        gradient_diff_mean_list = []
        gradient_diff_std_list = []
        grad_truth_norms_last_it_mean = []
        grad_truth_norms_last_it_std = []
        grad_truth_norms_min_mean = []
        grad_truth_norms_min_std = []

        for n in n_values:
            value_truths = []
            value_plugins = []
            grad_truths = []
            grad_plugins = []
            grad_truth_norms_last_it = []
            grad_truth_norms_min = []

            for sim in range(1, n_sim + 1):
                value_truth_file = os.path.join(dir, f"value_truth_trajectory_n{n}_sim{sim}_lam{lam}.npy")
                value_plugin_file = os.path.join(dir, f"value_plugin_trajectory_n{n}_sim{sim}_lam{lam}.npy")
                grad_truth_file = os.path.join(dir, f"grad_truth_trajectory_n{n}_sim{sim}_lam{lam}.npy")
                grad_plugin_file = os.path.join(dir, f"grad_plugin_trajectory_n{n}_sim{sim}_lam{lam}.npy")

                value_truths.append(np.load(value_truth_file)[0])
                value_plugins.append(np.load(value_plugin_file)[0])        
                grad_truths.append(np.load(grad_truth_file)[0])
                grad_plugins.append(np.load(grad_plugin_file)[0])
                grad_truth = np.load(grad_truth_file)
                grad_truth_norms_last_it.append(np.linalg.norm(grad_truth, axis=1)[-1])
                grad_truth_norms_min.append(np.min(np.linalg.norm(grad_truth, axis=1)))

            value_diff = np.array(value_truths) - np.array(value_plugins)
            value_diff_mean = np.mean(np.abs(value_diff))
            value_diff_std = np.std(np.abs(value_diff)) / np.sqrt(n_sim)
            value_diff_mean_list.append(value_diff_mean)
            value_diff_std_list.append(value_diff_std)
            
            grad_diff = np.array(grad_truths) - np.array(grad_plugins)
            grad_diff_norms = np.linalg.norm(grad_diff, axis=1)
            grad_diff_mean = np.mean(grad_diff_norms)
            grad_diff_std = np.std(grad_diff_norms) / np.sqrt(n_sim)
            gradient_diff_mean_list.append(grad_diff_mean)
            gradient_diff_std_list.append(grad_diff_std)

            grad_truth_norms_last_it_mean.append(np.mean(grad_truth_norms_last_it))
            grad_truth_norms_last_it_std.append(np.std(grad_truth_norms_last_it) / np.sqrt(n_sim))

            grad_truth_norms_min_mean.append(np.mean(grad_truth_norms_min))
            grad_truth_norms_min_std.append(np.std(grad_truth_norms_min) / np.sqrt(n_sim))

        value_diff_mean_array = np.array(value_diff_mean_list)
        value_diff_std_array = np.array(value_diff_std_list)
        reference_value = value_diff_mean_array[0]
        theoretical_value = 10 * np.sqrt(2) * reference_value / np.sqrt(np.array(n_values))

        gradient_diff_mean_array = np.array(gradient_diff_mean_list)
        gradient_diff_std_array = np.array(gradient_diff_std_list)
        reference_grad = gradient_diff_mean_array[0]
        theoretical_grad = 10 * np.sqrt(2) * reference_grad / np.sqrt(np.array(n_values))

        grad_truth_norms_last_it_mean_array = np.array(grad_truth_norms_last_it_mean)
        grad_truth_norms_last_it_std_array = np.array(grad_truth_norms_last_it_std)
        theoretical_last_it = 10 * np.sqrt(2) * grad_truth_norms_last_it_mean_array[0] / np.sqrt(np.array(n_values))

        grad_truth_norms_min_mean_array = np.array(grad_truth_norms_min_mean)
        grad_truth_norms_min_std_array = np.array(grad_truth_norms_min_std)
        theoretical_min_grad = 10 * np.sqrt(2) * grad_truth_norms_min_mean_array[0] / np.sqrt(np.array(n_values))

        ax1.loglog(n_values, value_diff_mean_array, label=label, linestyle='-', color=color_map[label])
        ax1.fill_between(n_values, value_diff_mean_array - 2*value_diff_std_array,
                                value_diff_mean_array + 2*value_diff_std_array,
                                color=color_map[label],
                                alpha=0.2)
    
        ax2.loglog(n_values, gradient_diff_mean_array, label=label, linestyle='-', color=color_map[label])
        ax2.fill_between(n_values, gradient_diff_mean_array - 2*gradient_diff_std_array,
                             gradient_diff_mean_array + 2*gradient_diff_std_array,
                             color=color_map[label],
                             alpha=0.2)

        ax3.loglog(n_values, grad_truth_norms_last_it_mean_array, label=label, linestyle='-', color=color_map[label])
        ax3.fill_between(n_values, grad_truth_norms_last_it_mean_array - 2*grad_truth_norms_last_it_std_array,
                             grad_truth_norms_last_it_mean_array + 2*grad_truth_norms_last_it_std_array,
                             color=color_map[label],
                             alpha=0.2)

        ax4.loglog(n_values, grad_truth_norms_min_mean_array, label=label, linestyle='-', color=color_map[label])
        ax4.fill_between(n_values, grad_truth_norms_min_mean_array - 2*grad_truth_norms_min_std_array,
                                grad_truth_norms_min_mean_array + 2*grad_truth_norms_min_std_array,
                                color=color_map[label],
                                alpha=0.2)

    ax1.loglog(n_values, theoretical_value, 'k-.')
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$|\mathcal{F}(\omega_0)-\widehat{\mathcal{F}}(\omega_0)|$')
    ax1.set_title("(a)")
    ax1.grid(True)

    ax2.loglog(n_values, theoretical_grad, 'k-.')
    ax2.set_xlabel(r'$n$')
    ax2.set_ylabel(r'$\|\nabla\mathcal{F}(\omega_0)-\widehat{\nabla\mathcal{F}}(\omega_0)\|$')
    ax2.set_title("(b)")
    ax2.grid(True)

    ax3.loglog(n_values, theoretical_last_it, 'k-.')
    ax3.set_xlabel(r'$n$')
    ax3.set_ylabel(r'$\|\nabla\mathcal{F}(\omega_T)\|$')
    ax3.set_title("(c)")
    ax3.grid(True)

    ax4.loglog(n_values, theoretical_min_grad, 'k-.')
    ax4.set_xlabel(r'$n$')
    ax4.set_ylabel(r'$\min_{i=0,\ldots,T}\|\nabla\mathcal{F}(\omega_i)\|$')
    ax4.set_title("(d)")
    ax4.grid(True)

    legend1_handles = [
        Line2D([0], [0], color='k', linestyle='-',  label='Gaussian kernel'),
        Line2D([0], [0], color='k', linestyle='-.', label='Theoretical slope'),
    ]
    leg1 = fig.legend(handles=legend1_handles, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(legend1_handles), frameon=False, borderaxespad=0.0, bbox_transform=fig.transFigure, fontsize=20)
    fig.gca().add_artist(leg1)

    legend2_handles = [
        Patch(color=gaussian_color, label='Gaussian dist'),
        Patch(color=student_colors[0], label=r'Student dist ($\nu=2.1$)'),
        Patch(color=student_colors[1], label=r'Student dist ($\nu=2.5$)'),
        Patch(color=student_colors[2], label=r'Student dist ($\nu=2.9$)'),
    ]
    fig.legend(handles=legend2_handles, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend2_handles), frameon=False, borderaxespad=0.0, bbox_transform=fig.transFigure, fontsize=20)
    fig.tight_layout(rect=[0, 0.005, 1, 1])

    save_path = os.path.join(folder_name, f'all_plots.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs=4, required=True, help="List of directories where the trajectories are stored")
    args = parser.parse_args()

    dir_list = args.dirs
    labels = [
        "Gaussian kernel -- Gaussian", 
        "Gaussian kernel -- Student (df=2.1)",
        "Gaussian kernel -- Student (df=2.5)",  
        "Gaussian kernel -- Student (df=2.9)"
        ]
    
    plot(dir_list, labels)
