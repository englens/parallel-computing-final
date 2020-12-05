#ifndef PARTICLE_FILTER_CPU_H_
#define PARTICLE_FILTER_CPU_H_

#include "helper_functions_cpu.h"

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
};



class ParticleFilter {
	
	// Number of particles to draw
	int num_particles; 
	
	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles
	std::vector<double> weights;
	
public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param M Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	void init(double x, double y, double theta, double std[]);

	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	void updateWeights(double sensor_range, double std_landmark[], std::vector<landmark_measurement> observations,
			Map map_landmarks);

	void resample();
	
	void write(std::string filename);
	
	const bool initialized() const {
		return is_initialized;
	}
};



#endif /* PARTICLE_FILTER_CPU_H_ */
