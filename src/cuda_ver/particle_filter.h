#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include "helper_functions.h"
#include <curand_kernel.h>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

class ParticleFilter {
	
public:

	// Flag, if filter is initialized
	bool is_initialized;
	
	// Set of current particles
    double* p_xs;
    double* p_ys;
    double* p_thetas;
    double* p_weights;
    
    // Copies for global mem
    double* p_xs_dev;
    double* p_ys_dev;
    double* p_thetas_dev;
    double* p_weights_dev;
    
    // Storage for vals used in some functions
    int *p_resample_new_inds;
    double *dev_error;
    
    // Rand states for particles -- each particle gets 1
    curandState* p_rand_states;
    
	// Constructor
	// @param M Number of particles
	ParticleFilter() : is_initialized(false) {}

	// Destructor
	~ParticleFilter() {
	    free(p_xs);
	    free(p_ys);
	    free(p_thetas);
	    free(p_weights);
	    HANDLE_ERROR(cudaFree(p_xs_dev));
	    HANDLE_ERROR(cudaFree(p_ys_dev));
	    HANDLE_ERROR(cudaFree(p_thetas_dev));
	    HANDLE_ERROR(cudaFree(p_weights_dev));
	    HANDLE_ERROR(cudaFree(dev_error));
	    HANDLE_ERROR(cudaFree(p_resample_new_inds));
	}

	void init();

	void prediction(double delta_t, double velocity, double yaw_rate);

	void updateWeights(std::vector<landmark_measurement> observations, Map map_landmarks);
	
	void resample();
	
	void getAvgError(double gt_x, double gt_y, double gt_theta, double* err_out);
	
	void write(std::string filename);
	
	bool initialized() {
		return is_initialized;
	}
};



#endif /* PARTICLE_FILTER_H_ */
