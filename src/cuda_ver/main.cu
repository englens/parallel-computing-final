/*
 * main.cpp
 * Reads in data and runs 2D particle filter.
 * Modified to work with mrclam data.
 */

#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

int main() {
    string odo_datafile = "data/R1_Converted/R1_Odometry.txt";
	string ground_truth_datafile = "data/R1_Converted/R1_Groundtruth.txt";
	string map_datafile = "data/R1_Converted/Landmark_Groundtruth.txt";
	string landmark_measurement_datafile = "data/R1_Converted/R1_Measurement.txt";
	string particle_output_file = "data/Output/run_0_particles.txt";
	
	
	
	double delta_t = 0.2; // time between timesteps
	
	//TODO: How do i calculate this? or do i just estimate (leave as-is)?
	//Update: totally gonna leave as-is
	double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [range [m], bearing [rad]]
	
	default_random_engine gen;
	normal_distribution<double> N_obs_range(0, sigma_landmark[0]);
	normal_distribution<double> N_obs_bearing(0, sigma_landmark[1]);
	double n_r, n_b;
	
	// Map Data read
	Map map;
	if (!read_landmark_groundtruth(map_datafile, map)) {
		cout << "Error: Could not open map file" << endl;
		return -1;
	}
	
	// Ground Truth Data read
	vector<ground_truth> gt;
	if (!read_gt_data(ground_truth_datafile, gt)) {
		cout << "Error: Could not open ground truth data file" << endl;
		return -1;
	}
	
	// Position/Odometry Data Read
	vector<odo_sample> odo_meas;
	if (!read_odometry_data(odo_datafile, odo_meas)) {
		cout << "Error: Could not open position/control measurement file" << endl;
		return -1;
	}
	
	// Landmark Measurement Data Reads
	vector<landmark_measurement> landmark_meas;
	if (!read_landmark_measurements(landmark_measurement_datafile, landmark_meas)) {
	    cout << "Error: Could not open Landmark Measurement file" << endl;
		return -1;
	}
	
	// Run particle filter!
	int num_time_steps = odo_meas.size();
	ParticleFilter pf;
	double total_error[3] = {0,0,0};
	double cum_mean_error[3] = {0,0,0};
	
	vector<landmark_measurement> curr_landmark_measures; // container for current timestep measurements
	
	double curr_time; // current time in sim; recalculated every step
	int latest_unchecked_meas = 0; // Index of next measurement to check when looking for new measurements to use
	
	// Only time the main loop
	int start = clock(); // Timer for measuring performace
	cout << "Running Particle Filter..." << endl;
	for(int i = 0; i < num_time_steps; ++i) {
	    curr_time = odo_meas[i].time;
	    
	    //cout << "Time step index: " << i << "  Time: " << curr_time << " # LM: " << curr_landmark_measures.size() << endl;
	    
	    curr_landmark_measures.clear();
	    while(latest_unchecked_meas < landmark_meas.size() && landmark_meas[latest_unchecked_meas].time <= curr_time){
	        curr_landmark_measures.push_back(landmark_meas[latest_unchecked_meas]);
	        latest_unchecked_meas++;
	    }
	    
	    // simulate the addition of noise to noiseless observation data
		// TODO: create noise vector beforehand in parallel
		vector<landmark_measurement> noisy_observations;
		landmark_measurement obs;
		for (int j = 0; j < curr_landmark_measures.size(); ++j) {
			n_r = N_obs_range(gen);
			n_b = N_obs_bearing(gen);
			obs = curr_landmark_measures[j];
			obs.range = obs.range + n_r;
			obs.bearing = obs.bearing + n_b;
			noisy_observations.push_back(obs);
		}
	    
	    
	    //If first time step, init particle_filter
		if (!pf.initialized()) {
			pf.init(); 
		}
		else {
			// Otherwise, Predict the vehicle's next state (noiseless).
			pf.prediction(delta_t, odo_meas[i-1].velocity, odo_meas[i-1].yawrate);
		}

		// Update the weights and resample
		// Only do this if theres actually new measurements to measure vs
		if(curr_landmark_measures.size() > 0){
            pf.updateWeights(noisy_observations, map);
		    pf.resample();
        }

        // Sample error and display
        double avg_error[3];
        pf.getAvgError(gt[i].x, gt[i].y, gt[i].theta, avg_error);
		for (int j = 0; j < 3; ++j) {
			total_error[j] += avg_error[j];
			cum_mean_error[j] = total_error[j] / (double)(i + 1);
			
		}
		// Print the cumulative weighted error
		cout << "Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;
	}
	
    // Done! now cleanup and time.
    int stop = clock();
    
    double runtime = (stop - start) / double(CLOCKS_PER_SEC);
	cout << "Runtime (sec): " << runtime << endl;
	
	// TODO output partile info, perhaps need to do every x seconds
	
	return 0;
}
	
