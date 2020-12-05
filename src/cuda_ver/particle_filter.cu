#include <random>
#include <numeric>
#include <algorithm>

#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

#include "particle_filter.h"

using namespace std;

const int num_particles = 512; //Set to multiple of warpsize; 128 256 512 1024
const double xmin = 0;
const double xmax = 6;
const double ymin = -6;
const double ymax = 6;
const double thetamin = 0;
const double thetamax = 359;
const double PARTICLE_LINEAR_STD = 0.3;
const double PARTICLE_ANGULAR_STD = 0.01;
const double LANDMARK_X_STD = 0.3;
const double LANDMARK_Y_STD = 0.3;

__global__ void init_pf_kernel(curandState *global_state, double *x, double *y, double *theta, double *weight){
    int id = threadIdx.x + blockIdx.x * blockDim.x; // might just do everything in 1 block
    
    // Setup uniform generators
    
    // State unique to every thread id
    curandState local_state;
    curand_init(1234, id, 0, &local_state);    //curand_init(seed, sequence, offset, *state)
    // state saved locally for faster access; only save global at end
    
    //curand_uniform_double comes out 0-1
    
    x[id] = xmin + curand_uniform_double(&local_state)*(xmax-xmin);
    y[id] = ymin + curand_uniform_double(&local_state)*(ymax-ymin);
    theta[id] = thetamin + curand_uniform_double(&local_state)*(thetamax-thetamin);
    //weight[id] = 1;
    global_state[id] = local_state;
}

void ParticleFilter::init() {
    // Allocate memory for particles
    // Host versions
    p_xs = (double *)calloc(num_particles, sizeof(double));
    p_ys = (double *)calloc(num_particles, sizeof(double));
    p_thetas = (double *)calloc(num_particles, sizeof(double));
    p_weights = (double *)calloc(num_particles, sizeof(double));
    
    //Device versions
    HANDLE_ERROR(cudaMalloc((void **)&p_xs_dev, num_particles*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&p_ys_dev, num_particles*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&p_thetas_dev, num_particles*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&p_weights_dev, num_particles*sizeof(double)));
    
    //Set weights to 1 -- TODO uncomment this if they stay set to 1
    HANDLE_ERROR(cudaMemset(p_weights_dev, 1.0, num_particles*sizeof(double)));
    
    //Allocate space for curand state vars
    HANDLE_ERROR(cudaMalloc((void **)&p_rand_states, num_particles*sizeof(curandState)));
    // Container for new indicies for resampling
    HANDLE_ERROR(cudaMalloc((void **)&p_resample_new_inds, num_particles*sizeof(int)));
    
    HANDLE_ERROR(cudaMalloc((void **)&dev_error, sizeof(double)*3));
    
    // Run init kernel to generate initial particle values
    init_pf_kernel<<<1, num_particles>>>(p_rand_states, p_xs_dev, p_ys_dev, p_thetas_dev, p_weights_dev);
    
    // Only need to run this once
    is_initialized = true;
}

__global__ void prediction_pf_kernel(curandState *global_state, double velocity, double delta_yaw, double delta_t,
                                double *x, double *y, double *theta) {
    int id = threadIdx.x + blockIdx.x * blockDim.x; // might just do everything in 1 block
    curandState local_state = global_state[id];
    
    //Put vars in local mem for faster access
    double old_x = x[id];
    double old_y = y[id];
    double old_theta = theta[id];
    
    // Predict new values based on odometry
    double new_x = old_x + velocity*delta_t*cos(old_theta);
    double new_y = old_y + velocity*delta_t*sin(old_theta);
    double new_theta = old_theta + delta_t*delta_yaw;
    
    // Add Gaussian noise to each
    // normal comes out with mean 0 and std 1
    // multiply by desired std then add desired mean to get the desired dist
    x[id] = curand_normal_double(&local_state)*PARTICLE_LINEAR_STD + new_x;
    y[id] = curand_normal_double(&local_state)*PARTICLE_LINEAR_STD + new_y;
    theta[id] = curand_normal_double(&local_state)*PARTICLE_ANGULAR_STD + new_theta;  
    global_state[id] = local_state;
}

void ParticleFilter::prediction(double delta_t, double velocity, double delta_yaw) {
    // Only update dev values since we dont need them on the host yet
    prediction_pf_kernel<<<1, num_particles>>>(p_rand_states, velocity, delta_yaw, delta_t, p_xs_dev, p_ys_dev, p_thetas_dev);
}


// Current strategy: repeat kernel for every measurement
// Might be too slow; possibly use some sort of grid-level parallel?
// Thou theres usually only 1 landmark so shouldnt matter
__global__ void updateWeights_pf_kernel(double *x, double *y, double *theta, double *weight, 
                                        double gt_x, double gt_y, double range, double bearing){
    int id = threadIdx.x + blockIdx.x * blockDim.x; // might just do everything in 1 block
    
    double my_x = x[id];
    double my_y = y[id];
    double my_theta = theta[id];
    double old_weight = weight[id];
    
    double landmark_x_observed = my_x + range*cos(bearing+my_theta);
    double landmark_y_observed = my_y + range*sin(bearing+my_theta);
    
    double num = exp(-0.5 * (pow((landmark_x_observed - gt_x), 2) / pow(LANDMARK_X_STD, 2) + pow((landmark_y_observed - gt_y), 2) / pow(LANDMARK_Y_STD, 2)));
    double denom = 2 * M_PI * LANDMARK_X_STD * LANDMARK_Y_STD;
    
    __shared__ double sum_weights[num_particles];
    double temp_local_weight = old_weight * num/denom;    
    sum_weights[id] = temp_local_weight;    
    __syncthreads();
    
    // Parallel reduction
    for (int size = num_particles/2; size>0; size/=2){
        if (id<size){
            sum_weights[id] += sum_weights[id+size];
        }
        __syncthreads();
    }
    // Now, sum_weights[0] is actual sum
    weight[id] = temp_local_weight/sum_weights[0];
}

// Update the weights of each particle using a multi-variate Gaussian distribution.
void ParticleFilter::updateWeights(std::vector<landmark_measurement> observations, Map map_landmarks) {

    //Reset weights to 1, to be reduced by following code
    HANDLE_ERROR(cudaMemset(p_weights_dev, 1.0, num_particles*sizeof(double)));
    for(int j=0; j<observations.size(); ++j){
        landmark_measurement curr_landmark_meas = observations[j];
        
        Map::single_landmark_s landmark_gt;
        bool found = false;
        
        //Select the landmark with the same id as curr_landmark (for gt)
        for(int k=0; k<map_landmarks.landmark_list.size(); k++){
            landmark_gt = map_landmarks.landmark_list[k];
            if (landmark_gt.id == curr_landmark_meas.id){
                found = true;
                break;
            }
        }
        if(!found){
            cout << "Error: Could not match observed id to landmark." << endl;
            return;
        }
        
        updateWeights_pf_kernel<<<1, num_particles>>>(p_xs_dev, p_ys_dev, p_thetas_dev, p_weights_dev, 
                                                      landmark_gt.x, landmark_gt.y, curr_landmark_meas.range, curr_landmark_meas.bearing);
                                                      
        // Weights now updated and normalized
    }
}


// Cant figure out how to do discrete_distribution on the device so we do next best option.
// Generate new particle indexes in host, then preform particle transfer on device
// NOTE does not resample weights. This should be fine tho, because weights are re-written 
__global__ void resample_pf_kernel(double *x, double *y, double *theta, double *weight, int *new_ind){
    int id = threadIdx.x + blockIdx.x * blockDim.x; // might just do everything in 1 block
    
    int src_ind = new_ind[id];
    double new_x;
    double new_y;
    double new_theta;
    double new_weight;
    
    new_x = x[src_ind];
    new_y = y[src_ind];
    new_theta = theta[src_ind];
    new_weight = weight[src_ind];
    __syncthreads();
    
    x[id] = new_x;
    y[id] = new_y;
    theta[id] = new_theta;
    weight[id] = new_weight;
}
// Unfortunatly must do two Memcpys in order to compute discrete vals on host
void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
    default_random_engine gen;

    // Grab weight vals from device
    // TODO This is the slowest part of this function -- can we do this all on gpu? (if only to not have to memcpy)
    HANDLE_ERROR(cudaMemcpy(p_weights, p_weights_dev, sizeof(double)*num_particles, cudaMemcpyDeviceToHost));
    
    //distribution requires an iterator, so we need an array type to get begin and end
    double p_weights2[num_particles];
    for (int i=0; i< num_particles; i++){
        p_weights2[i] = p_weights[i];
    }
    // Random integers on the [0, n) range
    // the probability of each particle is based on its weight
    //discrete_distribution<int> distribution(p_weights.begin(), p_weights.end());
    discrete_distribution<int> distribution(std::begin(p_weights2), std::end(p_weights2));
    int new_inds [num_particles];
    for (int i = 0; i < num_particles; i++){
        new_inds[i] = distribution(gen);
    }

    HANDLE_ERROR(cudaMemcpy(p_resample_new_inds, new_inds, sizeof(int)*num_particles, cudaMemcpyHostToDevice));
    resample_pf_kernel<<<1, num_particles>>>(p_xs_dev, p_ys_dev, p_thetas_dev, p_weights_dev, p_resample_new_inds);
}

// Interfaces with the device to calc the error
// Finds the particle with the lowest weight, then 
__global__ void error_pf_kernel(double *x, double *y, double *theta, double *weight,
                                   double gt_x, double gt_y, double gt_theta, double *err_out){
    int id = threadIdx.x + blockIdx.x * blockDim.x; // might just do everything in 1 block
    __shared__ int best_particle[num_particles/2];
    __shared__ double s_weights[num_particles];
    
    s_weights[id] = weight[id];
    
    __syncthreads();
    
    // Parallel reduction to find best particle, and store id in best_particle[0]
    for (int size = num_particles/2; size>0; size/=2){
        if (id<size){
            if (s_weights[id] < s_weights[id+size]){
                best_particle[id] = id+size;
                s_weights[id] = s_weights[id+size];
            }else{
                best_particle[id] = id;
                // keep s_weight
            }
        }
        __syncthreads();
    }
    //now best_particle[0] = id of best
    //s_weights[0] = weight of best
    //Do the rest on single thread
    if(id == 0){
        int best_ind = best_particle[0];
        double theta_err;
        err_out[0] = abs(x[best_ind] - gt_x);
        err_out[1] = abs(y[best_ind] - gt_y);

        theta_err = abs(theta[best_ind] - gt_theta);
        theta_err = (double)fmodf((float)theta_err, (float)2.0*M_PI);
        if (theta_err > M_PI) {
	        theta_err = 2.0 * M_PI - theta_err;
        }
        err_out[2] = theta_err;
    }
}
void ParticleFilter::getAvgError(double gt_x, double gt_y, double gt_theta, double *err_out) {
    
    error_pf_kernel<<<1, num_particles>>>(p_xs_dev, p_ys_dev, p_thetas_dev, p_weights_dev, gt_x, gt_y, gt_theta, dev_error);
    HANDLE_ERROR(cudaMemcpy(err_out, dev_error, sizeof(double)*3, cudaMemcpyDeviceToHost)); 
    
}

// NOTE Slow! Memcpy's data to host!
// Only run occasionally; maybe every second
// TODO time,weights
void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	HANDLE_ERROR(cudaMemcpy(p_xs, p_xs_dev, sizeof(double)*num_particles, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(p_ys, p_ys_dev, sizeof(double)*num_particles, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(p_thetas, p_thetas_dev, sizeof(double)*num_particles, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(p_weights, p_weights_dev, sizeof(double)*num_particles, cudaMemcpyDeviceToHost));
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << p_xs[i] << " " << p_ys[i] << " " << p_thetas[i] << "\n";
	}
	dataFile.close();
}
