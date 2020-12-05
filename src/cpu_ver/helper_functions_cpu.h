#ifndef HELPER_FUNCTIONS_CPU_H_
#define HELPER_FUNCTIONS_CPU_H_

#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

// Map is the collection of all landmarks
class Map {
public:
	
	struct single_landmark_s{

		int id; // Landmark ID
		float x; // Landmark x-position in the map
		float y; // Landmark y-position in the map
		float std_x; // Standard deviation of x measurements
		float std_y; // Standard deviation of y measurements
	};

	std::vector<single_landmark_s> landmark_list ; // List of landmarks in the map
};


/*
 * Struct representing one sample of odometry data.
 */
struct odo_sample {
	double time;        // (s)
	double velocity;	// Velocity [m/s]
	double yawrate;		    // Yaw [rad/s]
};

/*
 * Struct representing one ground truth position.
 */
struct ground_truth {
	double time;
	double x;		// Global vehicle x position [m]
	double y;		// Global vehicle y position
	double theta;	// Global vehicle yaw [rad]
};

/*
 * Struct representing one landmark observation measurement.
 */
struct landmark_measurement {
    
	int id;				    // Id of matching landmark in the map.
	double time;            // Time (s) of measurement (from beginning of data)
	double range;			// Range (m) from robot to landmark
	double bearing;			// Bearing φ (rad) from robot to landmark
};

//inline double dist(double x1, double y1, double x2, double y2) {
//	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
//}

inline double * getError(double gt_x, double gt_y, double gt_theta, double pf_x, double pf_y, double pf_theta) {
	static double error[3];
	error[0] = fabs(pf_x - gt_x);
	error[1] = fabs(pf_y - gt_y);
	error[2] = fabs(pf_theta - gt_theta);
	error[2] = fmod(error[2], 2.0 * M_PI);
	if (error[2] > M_PI) {
		error[2] = 2.0 * M_PI - error[2];
	}
	return error;
}

/* ------------------ DATA READ FUNCTIONS ------------------ */

//Reads groundtruth file from mrclam dataset.
//File format:
// <landmark_id>, <x>, <y>, <x_std>, <y_std> (all meters)
inline bool read_landmark_groundtruth(std::string filename, Map& map){   
    std::ifstream in_file(filename.c_str(), std::ifstream::in);
    
    if(!in_file){
        return false;
    }
    
    std::string curr_line;
    
    while(getline(in_file, curr_line)){
        std::istringstream iss_line(curr_line);
        
        float landmark_x, landmark_y, landmark_std_x, landmark_std_y;
        int id;       
        
        iss_line >> id;
        iss_line >> landmark_x;
        iss_line >> landmark_y;
        iss_line >> landmark_std_x;
        iss_line >> landmark_std_y;
        
        Map::single_landmark_s single_landmark_temp;
        
        single_landmark_temp.id = id;
        single_landmark_temp.x = landmark_x;
        single_landmark_temp.y = landmark_y;
        single_landmark_temp.std_x = landmark_std_x;
        single_landmark_temp.std_y = landmark_std_y;
        
        map.landmark_list.push_back(single_landmark_temp);
    }
    return true;
}

//data format: <time>, <velocity>, <yaw>
inline bool read_odometry_data(std::string filename, std::vector<odo_sample>& position_meas){

    std::ifstream in_file(filename.c_str(), std::ifstream::in);
    
    if (!in_file){
        return false;
    }
    
    std::string curr_line;
    while(getline(in_file, curr_line)){
        std::istringstream iss_pos(curr_line);
        
        double time, velocity, yawrate;
        
        odo_sample meas;
        
        iss_pos >> time;
        iss_pos >> velocity;
        iss_pos >> yawrate;
        
        meas.time = time;
        meas.velocity = velocity;
        meas.yawrate = yawrate; 
        
        position_meas.push_back(meas);
    }
    return true;
}

// Reads ground truth data from filename
// Format <time>, <x>, <y>, <theta/yaw>
inline bool read_gt_data(std::string filename, std::vector<ground_truth>& gt) {
    // Get file of position measurements:
	std::ifstream in_file(filename.c_str(),std::ifstream::in);
	// Return if we can't open the file.
	if (!in_file) {
		return false;
	}
	std::string curr_line;
	
	while(getline(in_file, curr_line)){
	    std::istringstream iss_line(curr_line);
	    
	    double time, x, y, theta;
	    
	    iss_line >> time;
	    iss_line >> x;
	    iss_line >> y;
	    iss_line >> theta;
	    
	    ground_truth single_gt;
	    
	    single_gt.time = time;
	    single_gt.x = x;
	    single_gt.y = y;
	    single_gt.theta = theta;
	    
	    gt.push_back(single_gt);
	}
	return true;
}


// Reads information about landmark *measurements*
// Format: <time>, <landmark_id>, <range>, <bearing>
inline bool read_landmark_measurements(std::string filename, std::vector<landmark_measurement>& observations) {
    std::ifstream in_file_meas(filename.c_str(), std::ifstream::in);
    
    if (!in_file_meas){
        return false;
    }
    
    std::string line_meas;
    
    // read lines from in_file_obs to line_obs
    while(getline(in_file_meas, line_meas)){
        std::istringstream iss_meas(line_meas);
        
        double meas_range, meas_bearing, meas_time;
        int landmark_id;
        
        iss_meas >> meas_time;
        iss_meas >> landmark_id;
        iss_meas >> meas_range;
        iss_meas >> meas_bearing; 
        
        //construct the landmark observation 
        landmark_measurement meas;
        
        meas.id = landmark_id;
        meas.time = meas_time;
        meas.range = meas_range;
        meas.bearing = meas_bearing;
        
        observations.push_back(meas);
    }
    return true;    
}

#endif /* HELPER_FUNCTIONS_CPU_H_ */
