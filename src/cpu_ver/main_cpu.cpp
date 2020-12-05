#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>

#include "particle_filter_cpu.h"
#include "helper_functions_cpu.h"

using namespace std;

int main() {
    string odo_datafile = "data/R1_Converted/R1_Odometry.txt";
	string ground_truth_datafile = "data/R1_Converted/R1_Groundtruth.txt";
	string map_datafile = "data/R1_Converted/Landmark_Groundtruth.txt";
	string landmark_measurement_datafile = "data/R1_Converted/R1_Measurement.txt";
	string particle_output_file = "data/Output/run_0_particles.txt";
	
	// parameters related to grading.
	int time_steps_before_lock_required = 100; // number of time steps before accuracy is checked by grader.
	double max_runtime = 45; // Max allowable runtime to pass [sec]
	double max_translation_error = 1; // Max allowable translation error to pass [m]
	double max_yaw_error = 0.05; // Max allowable yaw error [rad]
	
	double delta_t = 0.2; // time between timesteps
	double sensor_range = 50; // Sensor range [m]
	
	double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
	double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [range [m], bearing [rad]]
	
	default_random_engine gen;
	normal_distribution<double> N_x_init(0, sigma_pos[0]);
	normal_distribution<double> N_y_init(0, sigma_pos[1]);
	normal_distribution<double> N_theta_init(0, sigma_pos[2]);
	normal_distribution<double> N_obs_range(0, sigma_landmark[0]);
	normal_distribution<double> N_obs_bearing(0, sigma_landmark[1]);
	double n_x, n_y, n_theta, n_r, n_b;
	
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
	    
	    //If first time step, init particle_filter
		if (!pf.initialized()) {
			n_x = N_x_init(gen);
			n_y = N_y_init(gen);
			n_theta = N_theta_init(gen);
			
			pf.init(gt[i].x + n_x, gt[i].y + n_y, gt[i].theta + n_theta, sigma_pos); 
			
		}
		else {
			// Otherwise, Predict the vehicle's next state (noiseless).
			pf.prediction(delta_t, sigma_pos, odo_meas[i-1].velocity, odo_meas[i-1].yawrate);
		}
		
		// simulate the addition of noise to noiseless observation data.
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
        
		// Update the weights and resample
		pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);
		pf.resample();

        // Calculate and output the average weighted error of the particle filter over all time steps so far.
		vector<Particle> particles = pf.particles;
		int num_particles = particles.size();
		double highest_weight = 0.0;
		Particle best_particle;
		for (int i = 0; i < num_particles; ++i) {
			if (particles[i].weight > highest_weight) {
				highest_weight = particles[i].weight;
				best_particle = particles[i];
			}
		}
		double *avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, best_particle.x, best_particle.y, best_particle.theta);
		
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
	
	// TODO output partile info, perhaps need to gather in big table
	
	return 0;
}
	
