
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter_cpu.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 512;

    weights.resize(num_particles);
    particles.resize(num_particles);

    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // Normal distribution for x, y and theta
    normal_distribution<double> dist_x(x, std_x); // mean is centered around the new measurement
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    default_random_engine gen;

    // create particles and set their values
    for(int i=0; i<num_particles; ++i){
        Particle p;
        p.id = i;
        p.x = dist_x(gen); // take a random value from the Gaussian Normal distribution and update the attribute
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles[i] = p;
        weights[i] = p.weight;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double delta_yaw) {
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    default_random_engine gen;
    //std::cout << "P0.x: " << particles[0].x << " P0.y: " << particles[0].y << " P0.theta: "<< particles[0].theta << " PO.weight: " << weights[0] << std::endl;
    for(int i=0; i<num_particles; ++i){
        Particle *p = &particles[i]; // get address of particle to update
                
        // Havent actually verified if this works; using anyway
        double new_x = p->x + velocity*delta_t*cos(p->theta);
        double new_y = p->y + velocity*delta_t*sin(p->theta);
        double new_theta = p->theta + (delta_yaw*delta_t);

        // add Gaussian Noise to each measurement
        // Normal distribution for x, y and theta
        normal_distribution<double> dist_x(new_x, std_x);
        normal_distribution<double> dist_y(new_y, std_y);
        normal_distribution<double> dist_theta(new_theta, std_theta);

        // update the particle attributes
        p->x = dist_x(gen);
        p->y = dist_y(gen);
        p->theta = dist_theta(gen);
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<landmark_measurement> observations, Map map_landmarks) {
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double weights_sum = 0;

    for(int i=0; i<num_particles; ++i){
        Particle *p = &particles[i];
        double wt = 1.0;

        // convert observation from vehicle's to map's coordinate system
        for(int j=0; j<observations.size(); ++j){
        
            landmark_measurement curr_landmark = observations[j];
            
            double landmark_x_observed = p->x + curr_landmark.range*cos(curr_landmark.bearing+p->theta);
            double landmark_y_observed = p->y + curr_landmark.range*sin(curr_landmark.bearing+p->theta);
            int landmark_id = curr_landmark.id;

            
            
            Map::single_landmark_s landmark;
            bool found = false;
            //Select the landmark with the same id as curr_landmark
            for(int k=0; k<map_landmarks.landmark_list.size(); k++){
                landmark = map_landmarks.landmark_list[k];
                if (landmark.id == landmark_id){
                    found = true;
                    break;
                }
            }
            if(!found){
                cout << "Error: Could not match observed id to landmark." << endl;
                return;
            }
            
            double num = exp(-0.5 * (pow((landmark_x_observed - landmark.x), 2) / pow(std_x, 2) + pow((landmark_y_observed - landmark.y), 2) / pow(std_y, 2)));
            double denom = 2 * M_PI * std_x * std_y;
            wt *= num/denom;
        }
        weights_sum += wt;
        p->weight = wt;
    }
    // normalize weights to bring them in (0, 1]
    for (int i = 0; i < num_particles; i++) {
        Particle *p = &particles[i];
        p->weight /= weights_sum;
        weights[i] = p->weight;
    }
}

void ParticleFilter::resample() {
    default_random_engine gen;

    // Random integers on the [0, n) range
    // the probability of each individual integer is its weight of the divided by the sum of all weights.
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    vector<Particle> resampled_particles;

    for (int i = 0; i < num_particles; i++){
        resampled_particles.push_back(particles[distribution(gen)]);
    }

    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
