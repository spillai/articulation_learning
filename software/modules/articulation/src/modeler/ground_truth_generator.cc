#include <iomanip> 
#include <vector>
#include <map>
#include <iostream>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include <lcmtypes/articulation.hpp>
// #include <lcmtypes/articulation_object_pose_msg_t.h>
// #include <lcmtypes/articulation_object_pose_track_msg_t.h>
// #include <lcmtypes/articulation_object_pose_track_list_msg_t.h>

#include "lcmtypes/er_lcmtypes.h" 

#include <bot_core/bot_core.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_lcmgl_client/lcmgl.h>

#include <articulation/factory.h>
#include <articulation/utils.hpp>
#include <boost/foreach.hpp>
#include <ConciseArgs>

int varSAMPLES = 500; 
int varCLUSTER_SIZE = 10; 
double varSIGMA_POS = 0.f;
double varSIGMA_ANGLE = 0.f; 
std::string vLOG_FILENAME; 

using namespace std;
// using namespace articulation_models;

lcm_t *lcm = NULL;
cv::RNG rng;

typedef std::vector<articulation::pose_msg_t> Track;
std::vector<Track> tracks;

struct Pose { 
    double pos[3];
    double orientation[4];
    Pose() {
        pos[0] = 0, pos[1] = 0, pos[2] = 0;
        orientation[0] = 1, orientation[1] = 0, orientation[2] = 0, orientation[3] = 0;
    }
    Pose(const double* _pos, const double* _orientation) { 
        pos[0] = _pos[0], pos[1] = _pos[1], pos[2] = _pos[2];
        orientation[0] = _orientation[0], orientation[1] = _orientation[1], 
            orientation[2] = _orientation[2], orientation[3] = _orientation[3];
    }
};

struct DPose { 
    double pos[3];
    double rpy[3];
    DPose() {
        pos[0] = 0, pos[1] = 0, pos[2] = 0;
        rpy[0] = 0, rpy[1] = 0, rpy[2] = 0;
    }
    DPose(const double* _pos, const double* _rpy) { 
        pos[0] = _pos[0], pos[1] = _pos[1], pos[2] = _pos[2];
        rpy[0] = _rpy[0], rpy[1] = _rpy[1], rpy[2] = _rpy[2];
    }
};

std::map<int, int> parent;
std::map<int, Pose> initial_pose;
std::map<int, std::vector<Pose> > poses;
std::map<int, DPose> d_pose;
std::map<int, bool> done;

int getNextID() { 
    return initial_pose.size();
}

int getCurrentID() { 
    return initial_pose.size()-1;
}


void build_rigid_link(int id, int parent_id, double* pos, double* orientation) { 
    parent[id] = parent_id;
    done[id] = false;

    Pose p(pos, orientation);
    initial_pose[id] = p;

    double dpos[3] = {0,0,0};
    double drpy[3] = {0,0,0};
    d_pose[id] = DPose(dpos, drpy);
    
}

void build_prismatic_link(int id, int parent_id, double* pos, double* orientation, double* vec) { 
    parent[id] = parent_id;
    done[id] = false;

    Pose p(pos, orientation);
    initial_pose[id] = p;

    double dpos[3] = {vec[0],vec[1],vec[2]};
    double drpy[3] = {0,0,0};
    d_pose[id] = DPose(dpos, drpy);
    
    return;
}

void build_revolute_link(int id, int parent_id, double* pos, double* orientation, double* vec) { 
    parent[id] = parent_id;
    done[id] = false;

    Pose p(pos, orientation);
    initial_pose[id] = p;

    double dpos[3] = {0,0,0};
    double drpy[3] = {vec[0],vec[1],vec[2]};
    d_pose[id] = DPose(dpos, drpy);
    
    return;
}

void update_link(int id) { 
    if (parent[id] >= 0)
        if (!done[parent[id]])
            return;

    Pose p;
    if (parent[id] >= 0) { 
        Pose ip = initial_pose[id];
        Pose pp = initial_pose[parent[id]];

        double ip_mat[16], pp_mat[16];
        bot_quat_pos_to_matrix(ip.orientation, ip.pos, ip_mat);
        bot_quat_pos_to_matrix(pp.orientation, pp.pos, pp_mat);

        cv::Mat_<double> IPmat(4,4,&ip_mat[0]);
        cv::Mat_<double> PPmat(4,4,&pp_mat[0]);

        // std::cerr << "IPMat: " << IPmat << std::endl;
        // std::cerr << "PPMat: " << PPmat << std::endl;

        cv::Mat_<double> result = IPmat * PPmat;
        cv::Mat roi = result(cv::Rect(0,0,3,3));
        double* rot = (double*)roi.clone().data;

        bot_matrix_to_quat(rot, p.orientation);
        p.pos[0] = result(0,3);
        p.pos[1] = result(1,3);
        p.pos[2] = result(2,3);

        // std::cerr << "result: " << result << std::endl;

    } else { 
        p = initial_pose[id];
    }

        
    DPose& d = d_pose[id];

    Pose fp;

    double drpy[3];
    double dpos[3];

    articulation::pose_msg_t pose_msg;
    for (int j=0; j<varSAMPLES; j++) { 

        for (int k=0; k<3; k++) { 
            dpos[k] = d.pos[k] * j + rng.gaussian(varSIGMA_POS);
            drpy[k] = d.rpy[k] * j + rng.gaussian(varSIGMA_ANGLE);
        }            

        double p_mat[16], d_mat[16];
        bot_quat_pos_to_matrix(p.orientation, p.pos, p_mat);

        double dquat[4];
        bot_roll_pitch_yaw_to_quat(drpy, dquat);
        bot_quat_pos_to_matrix(dquat, dpos, d_mat);

        cv::Mat_<double> Pmat(4,4,&p_mat[0]);
        cv::Mat_<double> Dmat(4,4,&d_mat[0]);

        // std::cerr << "P: " << Pmat << std::endl;
        // std::cerr << "D: " << Dmat << std::endl;

        cv::Mat_<double> result = Dmat * Pmat;
        cv::Mat roi = result(cv::Rect(0,0,3,3));
        double* rot = (double*)roi.clone().data;

        bot_matrix_to_quat(rot, fp.orientation);
        fp.pos[0] = result(0,3);
        fp.pos[1] = result(1,3);
        fp.pos[2] = result(2,3);

        poses[id][j] = fp;
        
        pose_msg.utime = j;
        pose_msg.id = j;
        for (int k=0; k<3; k++)
            pose_msg.pos[k] = fp.pos[k];
        for (int k=0; k<4; k++)
            pose_msg.orientation[k] = fp.orientation[k];
        tracks[id].push_back(pose_msg);
    }

    done[id] = true;
    std::cerr << "done updating " << id << std::endl;
    return;
}

void update_tracks() { 
    tracks = std::vector<Track>(initial_pose.size());
    for (std::map<int, int>::iterator it=parent.begin(); it!=parent.end(); it++)
        poses[it->first] = std::vector<Pose>(varSAMPLES);
}

// int openChannel(articulation_object_pose_track_msg_t* track, std::string name, bool autocreate) { 
    
//     int j=0;
//     for (; j<track->num_channels; j++) { 
//         if (strcmp(track->channels[j].name, name.c_str()) == 0)
//             break;
//     }

//     if ( j == track->num_channels ) { 
//         if (!autocreate) return -1;
//         track->channels = (articulation_channelfloat32_msg_t*) realloc
//             (track->channels, sizeof(articulation_channelfloat32_msg_t)*(track->num_channels+1));
//         track->channels[j].name = strdup(name.c_str());
//         track->num_channels++;
//     }

//     track->channels[j].num_values = track->num_poses;
//     track->channels[j].values = (float*) malloc(sizeof(float) * track->channels[j].num_values );
//     return j;
        
// }

// void publish_tracks() { 
//     if (!tracks.size())
//         return;

//     int publish_count = 100;
//     while (publish_count <= varSAMPLES) { 
//         articulation_object_pose_track_list_msg_t pose_track_list_msg;
//         pose_track_list_msg.num_tracks = tracks.size();
//         pose_track_list_msg.tracks = (articulation_object_pose_track_msg_t*)
//             malloc(sizeof(articulation_object_pose_track_msg_t) * tracks.size());
//         pose_track_list_msg.tracks_projected = (articulation_object_pose_track_msg_t*)
//             malloc(sizeof(articulation_object_pose_track_msg_t) * tracks.size());
//         pose_track_list_msg.num_tracks_re = 0;
//         pose_track_list_msg.tracks_resampled = NULL; 
//         pose_track_list_msg.pose_flags = 1;
//         for (int j=0; j<tracks.size(); j++) { 
            
//             articulation_object_pose_track_msg_t& pose_track_msg = pose_track_list_msg.tracks[j];
//             pose_track_msg.id = j;
//             pose_track_msg.num_poses = publish_count;
//             pose_track_msg.num_poses_projected = 0;
//             pose_track_msg.num_poses_resampled = 0;
//             // pose_track_msg.num_channels = 0;
        
//             pose_track_msg.pose = (articulation_object_pose_msg_t*)
//                 malloc(sizeof(articulation_object_pose_msg_t) * publish_count);
//             pose_track_msg.pose_flags = (int32_t*)
//                 malloc(sizeof(int32_t) * publish_count);
//             pose_track_msg.pose_projected = NULL; 
//             pose_track_msg.pose_resampled = NULL; 
//             // pose_track_msg.channels = NULL;

//             // // open channels after num_poses are set
//             // int ch_width = openChannel(&pose_track_msg, "width", true);
//             // int ch_height = openChannel(&pose_track_msg, "width", true);

//             for (int k=0; k<tracks[j].size() && k < publish_count; k++) { 
//                 pose_track_msg.pose[k] = tracks[j][k];
//                 pose_track_msg.pose_flags[k] = ARTICULATION_OBJECT_POSE_TRACK_MSG_T_POSE_VISIBLE;
//                 // pose_track_msg.channels[ch_width].values[k] = 0.005;
//                 // pose_track_msg.channels[ch_height].values[k] = 0.005;
//             }
//             pose_track_list_msg.tracks_projected[j] = pose_track_list_msg.tracks[j];
//             articulation_object_pose_track_msg_t_publish(lcm, "ARTICULATION_OBJECT_POSE_TRACK", &pose_track_msg);

//             // free(pose_track_msg.pose);
//             // free(pose_track_msg.pose_flags);
//             //free(pose_track_msg.pose_channels);
//         }
//         articulation_object_pose_track_list_msg_t_publish(lcm, "ARTICULATION_OBJECT_POSE_TRACK_LIST", &pose_track_list_msg);

//         for (int j=0; j<tracks.size(); j++) 
//             free(pose_track_list_msg.tracks[j].pose), 
//                 free(pose_track_list_msg.tracks[j].pose_flags);                
//         free(pose_track_list_msg.tracks);
//         free(pose_track_list_msg.tracks_projected);

//         std::cerr << "Published " << publish_count << " tracks" << std::endl;
//         usleep(500000);
//         publish_count += 5;
//     }
//     publish_count = 100;
//     publish_tracks();

//     // Remeber to free
// }

void publish_tracks() { 
    if (!tracks.size())
        return;

    int publish_count = 100;
    while (publish_count <= varSAMPLES) { 
        erlcm_tracklet_list_t pose_track_list_msg;
        std::vector<erlcm_tracklet_t> _tracks(tracks.size());
        for (int j=0; j<tracks.size(); j++) { 

            _tracks[j].object_id = j; 
            _tracks[j].track_id = j;
            _tracks[j].num_poses = publish_count; 
            _tracks[j].poses = new bot_core_pose_t[publish_count];

            for (int k=0; k<tracks[j].size() && k < publish_count; k++) { 
                _tracks[j].poses[k].utime = k * 1e4;
                _tracks[j].poses[k].pos[0] = tracks[j][k].pos[0], 
                    _tracks[j].poses[k].pos[1] = tracks[j][k].pos[1], 
                    _tracks[j].poses[k].pos[2] = tracks[j][k].pos[2];
                _tracks[j].poses[k].orientation[0] = tracks[j][k].orientation[0], 
                    _tracks[j].poses[k].orientation[1] = tracks[j][k].orientation[1], 
                    _tracks[j].poses[k].orientation[2] = tracks[j][k].orientation[2],
                    _tracks[j].poses[k].orientation[3] = tracks[j][k].orientation[3];
            }
        }
        pose_track_list_msg.num_tracks = _tracks.size();
        pose_track_list_msg.tracks = &_tracks[0];
        erlcm_tracklet_list_t_publish(lcm, "TRACKLETS", &pose_track_list_msg);

        for (int j=0; j<_tracks.size(); j++) 
            free(_tracks[j].poses);

        std::cerr << "Published " << publish_count << " tracks" << std::endl;
        usleep(500000);
        publish_count += 5;
    }
    publish_count = 100;
    publish_tracks();

    // Remeber to free
}



void build_rigid_cluster(int clusters, int parent_id, double* p, double* q) { 

    // offset by p1
    for (int j=0; j<clusters; j++) {
        double p1[3] = { p[0] + rng.uniform(0.01f,.2f), 
                         p[1] + rng.uniform(0.01f,.2f), 
                         p[2] + rng.uniform(0.01f,.2f) };
        build_rigid_link(getNextID(), parent_id, p1, q);
    }
    return;
}    

void build_prismatic_cluster(int clusters, int parent_id, double* p, double* q, double* vec) { 

    // offset by p1
    for (int j=0; j<clusters; j++) {
        double p1[3] = { p[0] + rng.uniform(0.01f,.2f), 
                         p[1] + rng.uniform(0.01f,.2f), 
                         p[2] + rng.uniform(0.01f,.2f) };
        build_prismatic_link(getNextID(), parent_id, p1, q, vec);
    }
    return;
}    

void build_revolute_cluster(int clusters, int parent_id, double* p, double* q, double* vec) { 

    // offset by p1
    for (int j=0; j<clusters; j++) {
        double p1[3] = { p[0] + rng.uniform(0.1f,.2f), 
                         p[1] + rng.uniform(0.1f,.2f), 
                         p[2] + rng.uniform(0.1f,.2f) };
        build_revolute_link(getNextID(), parent_id, p1, q, vec);
    }
    return;
}    

void save_tracks() { 
     std::set<double> ids_set; 
     std::map<double, int> ids_map; 

     std::set<double> utimes_set;
     std::map<double, int> utimes_map;
    //--------------------------------------------
    // Determine ids, and utimes
    //--------------------------------------------
     { int idx = 0; 
         for (int j=0; j<tracks.size(); j++) { 
             ids_set.insert(double(j));
             ids_map[j] = idx++;

             for (int k=0; k<tracks[j].size(); k++) 
                 utimes_set.insert(k * 1e4);
         }
     }

    //--------------------------------------------
    // Determine ids, and utimes
    //--------------------------------------------
     { int idx = 0;
         for (std::set<double>::iterator it = utimes_set.begin(); 
              it != utimes_set.end(); it++) 
             utimes_map[*it] = idx++;
     }

     const int N = ids_map.size();
     const int T = utimes_map.size(); 
     const int D = 8; // 8 feats: u, v, x, y, z, nx, ny, nz
     if (!utimes_map.size() || !ids_map.size()) return;

    //--------------------------------------------
    // Fill data
    //--------------------------------------------
     int sz[] = {N, T, D};
     cv::SparseMat feats(3, &sz[0], CV_32F);

     int idx_[3];
     for (int j=0; j<tracks.size(); j++) { 
         const double feat_id = j;
         
         assert(ids_map.find(feat_id) != ids_map.end());
         int id_idx = ids_map[feat_id];
         assert(id_idx >=0 && id_idx < N);
         idx_[0] = id_idx; 
         
         const Track& track = tracks[j];
         for (int k=0; k<track.size(); k++) { 
             double utime = k * 1e4;
             assert(track[k].id == feat_id);
             assert(utimes_map.find(utime) != utimes_map.end());
             int utime_idx = utimes_map[utime];
             assert(utime_idx >=0 && utime_idx < T);

             // indices
             idx_[1] = utime_idx; idx_[2] = 0;
            
             // u,v
             feats.ref<float>(idx_) = 1; idx_[2]++;
             feats.ref<float>(idx_) = 1; idx_[2]++;

             // x,y,z
             feats.ref<float>(idx_) = track[k].pos[0]; idx_[2]++;
             feats.ref<float>(idx_) = track[k].pos[1]; idx_[2]++;
             feats.ref<float>(idx_) = track[k].pos[2]; idx_[2]++;

             // nx,ny,nz
             feats.ref<float>(idx_) = track[k].orientation[0]; idx_[2]++;
             feats.ref<float>(idx_) = track[k].orientation[1]; idx_[2]++;
             feats.ref<float>(idx_) = track[k].orientation[2];

         }
     }

     std::string fn = vLOG_FILENAME; 
     std::cerr << "Saving KLT Features to " << fn << std::endl;

     cv::FileStorage fs(fn, cv::FileStorage::WRITE);
     fs << "num_feats" << N;
     fs << "num_frames" << T;
     fs << "num_dims" << D;

     std::vector<double> ids(ids_set.begin(), ids_set.end());
     std::vector<double> utimes(utimes_set.begin(), utimes_set.end());
     cv::Mat_<double> ids_mat(ids);
     cv::Mat_<double> utimes_mat(utimes);


     fs << "feature_ids" << ids_mat; std::cerr << "ids: " << ids_mat << std::endl;
     fs << "feature_utimes" << utimes_mat; std::cerr << "utimes_mat: " << utimes_mat << std::endl;

     cv::Mat dfeats; feats.copyTo(dfeats);
     fs << "feature_data" << dfeats;
     fs.release();
     return;
}

int main(int argc, char** argv)
{
    lcm =  bot_lcm_get_global(NULL);

    ConciseArgs opt(argc, (char**)argv);
    opt.add(vLOG_FILENAME, "f", "log-file","Log file name");
    opt.add(varSIGMA_POS, "p", "sigma-pos", "Position Noise");
    opt.add(varSIGMA_ANGLE, "a", "sigma-angle", "Angular Noise");
    opt.add(varSAMPLES, "n", "samples", "No. of Samples");
    opt.add(varCLUSTER_SIZE, "c", "cluster-size", "No. of Trajectories per Cluster");
    opt.parse();

    rng = cv::RNG(2);

    double p[3]; 
    double q[4]; 
    double vec[3];

    // p[0] = rng.uniform(0.f,3.f), 
    //     p[1] = rng.uniform(0.f,3.f), 
    //     p[2] = rng.uniform(0.f,3.f);
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // build_rigid_cluster(5, -1, p, q);
 
    const int N = varCLUSTER_SIZE;

    p[0] = 0, p[1] = 0, p[2] = 0;
    q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    vec[0] = 0.001, vec[1] = 0, vec[2] = 0.003;
    build_prismatic_cluster(N, -1, p, q, vec);
    
    // 0-5
    p[0] = 0, p[1] = 0, p[2] = .5;
    q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    vec[0] = .00, vec[1] = 0.0, vec[2] = 0.006;
    build_revolute_cluster(N, 0, p, q, vec);

    // 5-10
    p[0] = .3, p[1] = 0, p[2] = 0;
    q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    vec[0] = .00, vec[1] = 0.0, vec[2] = 0.003;
    build_revolute_cluster(N, getCurrentID(), p, q, vec);

    // #mar15
    p[0] = 1, p[1] = 0, p[2] = 1;
    q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    vec[0] = 0.f, vec[1] = 0.0, vec[2] = 0.001;
    build_prismatic_cluster(N, 0, p, q, vec);

    // // #mar15
    // p[0] = 0, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = 0.001, vec[1] = 0, vec[2] = 0;
    // build_prismatic_cluster(N, -1, p, q, vec);

    // // 
    // p[0] = .3, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .00, vec[1] = 0.0, vec[2] = 0.003;
    // build_revolute_cluster(2*N, getNextID()-2, p, q, vec);

    // //
    // p[0] = .3, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .00, vec[1] = 0.0, vec[2] = 0.003;
    // build_revolute_cluster(N/2, getNextID()-2, p, q, vec);

    // p[0] = .5, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .00, vec[1] = 0.003, vec[2] = 0.003;
    // build_revolute_cluster(2*N, -1, p, q, vec);

    // // #mar15
    // p[0] = .5, p[1] = 0, p[2] = .50;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .003, vec[1] = 0.0, vec[2] = 0.f;
    // build_revolute_cluster(N, -1, p, q, vec);

    // p[0] = 0, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = 0.001, vec[1] = 0, vec[2] = 0.003;
    // build_prismatic_cluster(N/4, -1, p, q, vec);

    // p[0] = 1, p[1] = 0, p[2] = 1;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = 0.f, vec[1] = 0.0, vec[2] = 0.001;
    // build_prismatic_cluster(N/2, -1, p, q, vec);

    // p[0] = 0, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = 0.001, vec[1] = 0, vec[2] = 0;
    // build_prismatic_cluster(N, -1, p, q, vec);

    // p[0] = .5, p[1] = 0, p[2] = 0;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .009, vec[1] = 0.0, vec[2] = 0;
    // build_revolute_cluster(3*N/8, -1, p, q, vec);

    // p[0] = .5, p[1] = 0, p[2] = .50;
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // vec[0] = .00, vec[1] = 0.0, vec[2] = 0.003;
    // build_revolute_cluster(5*N/8, -1, p, q, vec);


    // rng = cv::RNG(3);
    // p[0] = rng.uniform(0.f,3.f), 
    //     p[1] = rng.uniform(0.f,3.f), 
    //     p[2] = rng.uniform(0.f,3.f);
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // build_rigid_cluster(5, -1, p, q);

    // rng = cv::RNG(4);
    // p[0] = rng.uniform(0.f,3.f), 
    //     p[1] = rng.uniform(0.f,3.f), 
    //     p[2] = rng.uniform(0.f,3.f);
    // q[0] = 1, q[1] = 0, q[2] = 0, q[3] = 0;
    // build_rigid_cluster(5, -1, p, q);

    // double p1[3] = { 0,0,0 };    
    // double q1[4] = { 1,0,0,0 };

    // build_rigid_link(getNextID(), -1, p1, q1);

    // // double p2[3] = { 0, 0, 0 } ;
    // // double vec2[3] = { 0.0005, 0, 0 }; 
    // // double q2[4] = { 1,0,0,0 };
    
    // // build_prismatic_link(getNextID(), 0, p2, q2, vec2);

    // double p3[3] = { .3, 0, 0 } ;
    // double vec3[3] = { 0, 0.0034, 0 }; 
    // double q3[4] = { 1,0,0,0 };
    
    // build_revolute_link(getNextID(), 0, p3, q3, vec3);

    // double p4[3] = { .2, 0, 0 } ;
    // double vec4[3] = { 0, 0.0034, 0 }; 
    // double q4[4] = { 1,0,0,0 };
    
    // build_revolute_link(getNextID(), 1, p4, q4, vec4);


    // double p5[3] = { .2, 0, 0 } ;
    // double vec5[3] = { 0, 0.0034, 0 }; 
    // double q5[4] = { 1,0,0,0 };
    
    // build_revolute_link(getNextID(), 1, p5, q5, vec5);


    update_tracks();
    
    bool all_done = false;
    while (!all_done) {

        for (std::map<int, int>::iterator it = parent.begin(); it != parent.end(); it++) { 
            std::cerr << "updating: " << it->first << std::endl;
            // if (it->second < 0) { 
                update_link(it->first);
                // }
        }
        
        bool d = true;
        for (std::map<int, bool>::iterator it=done.begin(); it != done.end(); it++)
            d = d && it->second;

        all_done = d;
    }

    save_tracks();
    publish_tracks();

    lcm_destroy(lcm);
    return 0;
}


