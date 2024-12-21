import matplotlib.pyplot as plt
import os
import numpy as np
import scienceplots
plt.style.use(['science','ieee', 'no-latex'])
plt.rcParams.update({'font.size': 12, 'hatch.linewidth': 0.25, 'hatch.color': 'gray', 'font.serif': 'DejaVu Sans',})

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def process_and_plot_txt_file(file_path):
    """Reads a text file and plots the data."""
    timestamps = []
    requests = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            try:
                _, num_requests = map(int, line.split())
                timestamps.append(i + 1)  # Assuming each line represents one second
                requests.append(num_requests)
            except ValueError:
                print(f"Skipping invalid line in file {file_path}: {line}")
                continue

    # the final one is incorrectly 0 for some reason, so remove it. Maybe because of final newline trace file?
    requests.pop()

    print(file_path)
    print(f'min: {min(requests)}')
    print(f'max: {max(requests)}')
    print(f'avg: {np.average(requests)}')
    print(f'std: {np.std(requests)}')

    # window_size = 1  # Adjust the window size for smoothing
    # smoothed_requests = np.convolve(requests, np.ones(window_size) / window_size, mode='valid')

    # # Generate timestamps for smoothed data
    # timestamps = np.arange(len(smoothed_requests))

    # # Plot the data
    # # fig, ax = plt.subplots(figsize=(3.3*1, 2.5))
    # fig, ax = plt.subplots(figsize=(3.3*1, 2.5))
    # ax.set_xticks([0, 86400, 86400*2, 86400*3, 86400*4, 86400*5])
    # ax.set_xticklabels(['0', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'])
    # plt.plot(timestamps, smoothed_requests, marker=None, linestyle='-', label="Requests per second")
    # # plt.title(f"Requests Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Number of Requests")
    # # plt.locator_params(axis='x', nbins=2)
    # # plt.locator_params(axis='y', nbins=3)

    # # plt.show()
    # plt.ylim(0, 150)
    # plt.grid(True)
    # plt.legend()

#
# ax.plot(x, y, marker='o')


    # Save the figure with the same name as the file (but as a .png)
    figure_name = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
    plt.savefig(f'{BASE_DIR}/results/{figure_name}')
    plt.close()
    print(f"Figure saved as: {figure_name}")


def traverse_directory_and_plot(directory):
    """Traverse a directory and process all .txt files."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                process_and_plot_txt_file(file_path)


# Main function
if __name__ == "__main__":
    directory_to_search = f"{BASE_DIR}/../client/trace"  # Replace with the target directory
    # traverse_directory_and_plot(directory_to_search)
    process_and_plot_txt_file(f"{BASE_DIR}/../client/trace/train_aio/train_aio.txt")
    process_and_plot_txt_file(f"{BASE_DIR}/../client/trace/test/shift_0h/req_over_time_2024-05-10-shift-0h.txt")
    process_and_plot_txt_file(f"{BASE_DIR}/../client/trace/val/req_over_time_2024-05-13.txt")
