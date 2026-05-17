[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CjRQqtHi)

# DIPPID Data Gatherer

This Python script captures accelerometer and gyroscope data from a DIPPID device over a UDP connection. It records data for exactly 10 seconds upon a button press, and then uses Pandas to resample the captured data to a precise 100Hz frequency before saving it to a structured CSV file.

## Requirements

Ensure you have Python installed along with the following dependencies:
* `pandas`
* `DIPPID`
* `sklearn`

You can install Pandas via pip if you haven't already:
```bash
pip install pandas
```
*(Note: Make sure the `DIPPID` module is correctly installed or located in your working directory).*

## How to Use

1. **Run the script:**
   ```bash
   python capturing_data.py
   
```
2. **Enter Metadata:** The script will prompt you in the terminal to enter your name, the activity (e.g., running, lifting), and a trial number. This determines the name of your output file.
3. **Start Recording:** Ensure your DIPPID app/device is connected. Press **Button 1** on the device to trigger the 10-second data capture.
4. **Perform Activity:** The script will capture data as fast as possible for the 10-second duration.
5. **Data Processing:** Once the time is up, the script automatically resamples the data to a steady 100Hz (averaging rapid captures and interpolating any gaps) and saves the CSV file to your current directory.

## Output Format

The output is a CSV file named `[user_name]-[activity]-[trial_num].csv`. 

It will contain exactly 1,000 rows (representing 10 seconds of data at 100Hz) with the following ordered columns:
* `id`: Sequential row index (0 to 999)
* `timestamp`: Elapsed time in seconds (e.g., 0.01, 0.02)
* `acc_x`, `acc_y`, `acc_z`: Accelerometer values
* `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope values

## Configuration

If you need to adjust the networking or timing, you can easily modify the global variables at the top of the script:
* `PORT`: Currently set to `5700`.
* `DURATION`: Currently set to `10.0` seconds.

Additionally, we subsample, so it always results in 1000 entries.
