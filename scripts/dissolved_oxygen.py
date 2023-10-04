import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_dissolved_oxygen(T_celsius):
    Twat_k = T_celsius + 273.15
    Ox_stat = np.exp(
        -139.3441
        + (1.575701 * 10**5 / Twat_k)
        - (6.642308 * 10**7 / Twat_k**2)
        + (1.243800 * 10**10 / Twat_k**3)
        - (8.621949 * 10**11 / Twat_k**4)
    )
    return Ox_stat


def calculate_and_save_dissolved_oxygen(input_csv_path: str) -> str:
    """
    Reads the CSV file from the given path, calculates the dissolved oxygen based on the river water temperatures,
    and then writes a new CSV file containing the dissolved oxygen values along with all other columns.

    Parameters:
        input_csv_path (str): Path to the input CSV file containing river water temperatures.

    Returns:
        str: Path to the output CSV file with dissolved oxygen values added.
    """

    df = pd.read_csv(input_csv_path)

    df["Ox_stat"] = df["avg_water_temp"].apply(compute_dissolved_oxygen)

    output_path = input_csv_path.replace(".csv", "_with_dissolved_oxygen.csv")
    df.to_csv(output_path, index=False)

    return output_path

def plot_do_timeseries(output_file_path):
    df = pd.read_csv(output_file_path)
    plt.figure(figsize=(20,10))
    x = df.date
    y = df.Ox_stat
    plt.plot(x, y, color = 'lightgrey', label = 'dissolved oxygen')
    plt.xlabel('Date', fontsize='16', weight='bold')
    plt.ylabel('Dissolved Oxygen (mg/L)', fontsize='16', weight='bold')
    _ = plt.xticks(x[::120],rotation=45,  fontsize='14')
    _ = plt.yticks(fontsize='14')
    plt.legend(fontsize = '12')
    plt.title('Timeseries Plot', fontsize = '18')
    plt.savefig('static/DO_timeseries.png')
    

#
# # Test the function
# output_file_path = calculate_and_save_dissolved_oxygen("./predicted_wt.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the CSV file as an argument.")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = calculate_and_save_dissolved_oxygen(input_file_path)
    print(f"Processed file saved to: {output_file_path}")
    plot_do_timeseries(output_file_path)
