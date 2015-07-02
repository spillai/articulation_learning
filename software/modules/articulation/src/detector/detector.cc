// Tag Detector
// Publish Tracks (one message per track/ID)
// articulation-detector -l ~/data/2013-07-31-drawer-apriltags/top_drawer_motion10fps -e 100 -k 10

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <lcmtypes/bot_core.hpp>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// April tags detector and various families that can be selected by command line option
#include <AprilTags/TagDetector.h>
#include <AprilTags/Tag16h5.h>
#include <AprilTags/Tag25h7.h>
#include <AprilTags/Tag25h9.h>
#include <AprilTags/Tag36h9.h>
#include <AprilTags/Tag36h11.h>

// lcm log player wrapper
#include <lcm-utils/lcm-reader-util.hpp>
#include <pcl-utils/pose_utils.hpp>

// lcm messages
#include <lcmtypes/articulation.hpp> 

// optargs
#include <ConciseArgs>

// visualization msgs
#include <lcmtypes/visualization.h>
#include <lcmtypes/visualization.hpp>

using namespace std;
using namespace cv;

// int unique_pose_id = -1;

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}

//----------------------------------
// State representation
//----------------------------------
struct state_t {
    lcm::LCM lcm;

    BotParam   *b_server;
    BotFrames *b_frames;

    cv::VideoWriter writer;

    AprilTags::TagDetector* m_tagDetector;
    AprilTags::TagCodes m_tagCodes;

    std::map<uint64_t, articulation::track_msg_t> tracks;
    std::map<uint64_t, int> id2idx_map;

    int m_width; // image size in pixels
    int m_height;
    double m_tagSize; // April tag side length in meters of square black frame
    double m_fx; // camera focal length in pixels
    double m_fy;
    double m_px; // camera principal point
    double m_py;


    state_t() : 
        b_server(NULL),
        b_frames(NULL), 
                
        m_tagDetector(NULL),
        m_tagCodes(AprilTags::tagCodes36h11), 

        m_width(640),
        m_height(480),
        m_tagSize(0.166),
        m_fx(600),
        m_fy(600),
        m_px(m_width/2),
        m_py(m_height/2) 
    {
        //----------------------------------
        // Bot Param/frames init
        //----------------------------------
        b_server = bot_param_new_from_server(lcm.getUnderlyingLCM(), 1);
        b_frames = bot_frames_get_global (lcm.getUnderlyingLCM(), b_server);

        //----------------------------------
        // Setup tag detector
        //----------------------------------
        m_tagDetector = new AprilTags::TagDetector(m_tagCodes);

    }

    ~state_t() { 
        if (m_tagDetector) delete m_tagDetector;
    }

    void* ptr() { 
        return (void*) this;
    }

    void publish_batch();

    void on_kinect_image_frame (const lcm::ReceiveBuffer* rbuf, 
                                const std::string& chan,
                                const kinect::frame_msg_t *msg);
    
    // void print_detection(AprilTags::TagDetection& detection) {
    //     cout << "  Id: " << detection.id
    //          << " (Hamming: " << detection.hammingDistance << ")";

    //     // recovering the relative pose of a tag:

    //     // NOTE: for this to be accurate, it is necessary to use the
    //     // actual camera parameters here as well as the actual tag size
    //     // (m_fx, m_fy, m_px, m_py, m_tagSize)

    //     Eigen::Vector3d translation;
    //     Eigen::Matrix3d rotation;
    //     detection.getRelativeTranslationRotation(m_tagSize, m_fx, m_fy, m_px, m_py,
    //                                              translation, rotation);

    //     Eigen::Matrix3d F;
    //     F <<
    //         1, 0,  0,
    //         0,  -1,  0,
    //         0,  0,  1;
    //     Eigen::Matrix3d fixed_rot = F*rotation;
    //     double yaw, pitch, roll;
    //     wRo_to_euler(fixed_rot, yaw, pitch, roll);

    //     cout << "  distance=" << translation.norm()
    //          << "m, x=" << translation(0)
    //          << ", y=" << translation(1)
    //          << ", z=" << translation(2)
    //          << ", yaw=" << yaw
    //          << ", pitch=" << pitch
    //          << ", roll=" << roll
    //          << endl;

        
    //     // Detection
    //     vs::obj_collection_t objs_msg; 
    //     objs_msg.id = 10000; 
    //     objs_msg.name = "POSE_EST | POSE_LIST"; 
    //     objs_msg.type = VS_OBJ_COLLECTION_T_AXIS3D; 
    //     objs_msg.reset = true; 
        
    //     vs::obj_t pose; 
    //     pose.id = detection.id; 
    //     pose.x = translation(0), pose.y = translation(1), pose.z = translation(2);
    //     pose.roll = roll, pose.pitch = pitch, pose.yaw = yaw; 

    //     objs_msg.objs.push_back(pose);
    //     objs_msg.nobjs = objs_msg.objs.size(); 

    //     lcm.publish("OBJ_COLLECTION", &objs_msg);
    // }

};
state_t state;

struct AprilTagsDetectorOptions { 
    bool vLIVE_MODE;
    bool vCREATE_VIDEO;
    bool vDEBUG;
    int vPLOT, vSTART_FRAME, vEND_FRAME, vPUBLISH_EVERY_K_POSES;
    float vMAX_FPS;
    float vSCALE;
    std::string vLOG_FILENAME;
    std::string vCHANNEL;

    AprilTagsDetectorOptions () : 
        vLOG_FILENAME(""), vCHANNEL("KINECT_FRAME"), 
        vMAX_FPS(30.f) , vSTART_FRAME(0), vEND_FRAME(-1), vPUBLISH_EVERY_K_POSES(1), 
        vSCALE(1.f), vDEBUG(true), vLIVE_MODE(false), vCREATE_VIDEO(false), vPLOT(1) {}
};
AprilTagsDetectorOptions options;


// LCM Log player wrapper
LCMLogReaderOptions poptions;
LCMLogReader player;

int openChannel(articulation::track_msg_t& track, std::string name, bool autocreate) { 
    
    int j=0;
    for (; j<track.num_channels; j++) { 
        if (track.channels[j].name == name)
            break;
    }
    if ( j == track.num_channels ) { 
        if (!autocreate) return -1;
        articulation::channelfloat32_msg_t ch;
        ch.name = name;
        track.channels.push_back(ch);
    }

    track.channels[j].values.resize(track.pose.size());
    track.channels[j].num_values = track.channels[j].values.size();
    return j;
        
}

pose_utils::pose_t construct_tf(Vec3f& p1, Vec3f& p2, Vec3f& p3, Vec3f& p4) { 
  cv::Vec3f x1 = p4-p3; // p1-p2;
    float x1norm = cv::norm(x1);
    x1 /= x1norm;

    cv::Vec3f y1 = p2-p3; // p3-p2;
    float y1norm = cv::norm(y1);
    y1 /= y1norm;

    cv::Vec3f z1 = x1.cross(y1);
    float z1norm = cv::norm(z1);
    z1 /= z1norm;

    y1 = z1.cross(x1);
    y1norm = cv::norm(y1);
    y1 /= y1norm;    

    double T1[9];
    T1[0] = x1[0], T1[1] = x1[1], T1[2] = x1[2];
    T1[3] = y1[0], T1[4] = y1[1], T1[5] = y1[2];
    T1[6] = z1[0], T1[7] = z1[1], T1[8] = z1[2];

    // Position, Orientation needs to be UKF mean
    pose_utils::pose_t pose; 
    pose.utime = 0;
    pose.set_translation(p3[0],p3[1],p3[2]);
    pose.set_rotation(T1);
    return pose;
}

static 
void populate_viz_cloud(vs::point3d_list_t& viz_list, const cv::Mat_<Vec3f>& cloud, const cv::Mat& img, 
                        int64_t object_id = bot_timestamp_now()) { 

    viz_list.nnormals = viz_list.normals.size(); 
    viz_list.npointids = viz_list.pointids.size(); 

    viz_list.id = bot_timestamp_now(); 
    viz_list.collection = 100; 
    viz_list.element_id = 1; 

    viz_list.npoints = cloud.rows * cloud.cols;
    viz_list.points = std::vector<vs::point3d_t>(cloud.rows * cloud.cols);

    viz_list.ncolors = img.rows * img.cols; 
    viz_list.colors = std::vector<vs::color_t>(img.rows * img.cols);

    const Vec3f* pptr = cloud.ptr<Vec3f>(0);
    const Vec3b* cptr = img.ptr<Vec3b>(0);
    for (int k=0; k<cloud.rows * cloud.cols; k++) { 
        viz_list.points[k].x = (*pptr)[0];
        viz_list.points[k].y = (*pptr)[1];
        viz_list.points[k].z = (*pptr)[2];

        viz_list.colors[k].r = (*cptr)[2] * 1.f / 255.f;
        viz_list.colors[k].g = (*cptr)[1] * 1.f / 255.f;
        viz_list.colors[k].b = (*cptr)[0] * 1.f / 255.f;

        pptr++;
        cptr++;
    }
    
    return;
}


void publish_cloud(state_t* state, const cv::Mat_<Vec3f>& cloud, const cv::Mat& img) { 

    //----------------------------------
    // Viz inits
    //----------------------------------
    vs::point3d_list_collection_t viz_cloud_msg;
    viz_cloud_msg.id = 1050; 
    viz_cloud_msg.name = "KINECT DEBUG"; 
    viz_cloud_msg.type = VS_POINT3D_LIST_COLLECTION_T_POINT; 
    viz_cloud_msg.reset = true; 

    viz_cloud_msg.point_lists.resize(1);
    populate_viz_cloud(viz_cloud_msg.point_lists[0], cloud, img);

    viz_cloud_msg.nlists = viz_cloud_msg.point_lists.size();
    state->lcm.publish("POINTS_COLLECTION", &viz_cloud_msg);

    return;

}

void publish_camera_frame(state_t* state, double utime) {
    //----------------------------------
    // Publish camera frame of reference for drawing
    //----------------------------------
    vs::obj_collection_t objs_msg; 
    objs_msg.id = 100; 
    objs_msg.name = "KINECT_POSE"; 
    objs_msg.type = VS_OBJ_COLLECTION_T_AXIS3D; 
    objs_msg.reset = true; 

    BotTrans sensor_frame;
    bot_frames_get_trans_with_utime (state->b_frames, "KINECT", "local", utime, &sensor_frame);
    double rpy[3]; bot_quat_to_roll_pitch_yaw(sensor_frame.rot_quat, rpy);
        
    objs_msg.objs.resize(1); 
    objs_msg.objs[0].id = 1; 
    objs_msg.objs[0].x = sensor_frame.trans_vec[0], 
        objs_msg.objs[0].y = sensor_frame.trans_vec[1], objs_msg.objs[0].z = sensor_frame.trans_vec[2]; 
    objs_msg.objs[0].roll = rpy[0], objs_msg.objs[0].pitch = rpy[1], objs_msg.objs[0].yaw = rpy[2]; 
    objs_msg.nobjs = objs_msg.objs.size(); 

    state->lcm.publish("OBJ_COLLECTION", &objs_msg);
    return;
}

pose_utils::pose_t transform_to_sensor_frame(state_t* state, double utime, 
                                          const pose_utils::pose_t& tag_pose) { 
    BotTrans sensor_frame;
    bot_frames_get_trans_with_utime (state->b_frames, "KINECT", "local", utime, &sensor_frame);

    pose_utils::pose_t sensor_pose(sensor_frame);
    pose_utils::pose_t tag_pose_tf = sensor_pose * tag_pose;
    return tag_pose_tf;
}

static pose_utils::pose_t compute_tag_pose(state_t* state, 
                                        double utime, 
                                        AprilTags::TagDetection& detection, 
                                        cv::Mat_<Vec3f>& cloud) { 

    // use corner points detected by line intersection
    std::pair<float, float> p1 = detection.p[0];
    std::pair<float, float> p2 = detection.p[1];
    std::pair<float, float> p3 = detection.p[2];
    std::pair<float, float> p4 = detection.p[3];

    pose_utils::pose_t tag_pose = construct_tf(cloud(p1.second,p1.first), 
                                            cloud(p2.second,p2.first), 
                                            cloud(p3.second,p3.first), 
                                            cloud(p4.second,p4.first));
    // std::cerr << "id: " << detection.id << std::endl;

    return tag_pose;
}

void state_t::publish_batch() { 
    bool print_once = true;

    articulation::track_list_msg_t track_list_msg;
    for (std::map<uint64_t, articulation::track_msg_t>::iterator it = tracks.begin(); 
         it != tracks.end(); it++) { 
        articulation::track_msg_t& track = it->second;
        std::cerr << "=====> Published: " 
                  <<"id="<<it->first<<":"<<track.pose.size() << std::endl;
        // if (track.pose.size() % options.vPUBLISH_EVERY_K_POSES == 0) { 
        //     lcm.publish("ARTICULATION_OBJECT_TRACKS", &track);
        //     if (print_once) { 
        
        //         print_once = false;
        //     }
        //     std::cerr <<"id="<<it->first<<":"<<track.pose.size() << std::endl;
        // }
        // std::cerr << "Track size for ID: " << it->first << ": " << track.pose.size() << std::endl;
        if (track.pose.size() >= 100) { 
            track.pose.erase(track.pose.begin(), track.pose.end() - 100); 
            track.pose_flags.erase(track.pose_flags.begin(), track.pose_flags.end() - 100); 
            track.num_poses = track.pose.size();
        }
        track_list_msg.tracks.push_back(track);
    }

    track_list_msg.num_tracks = track_list_msg.tracks.size();
    lcm.publish("ARTICULATION_OBJECT_TRACKS", &track_list_msg);
}

static void on_kinect_image_frame(const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                                   const kinect::frame_msg_t *msg) {
    state.on_kinect_image_frame(rbuf, chan, msg);
}

void state_t::on_kinect_image_frame (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                                   const kinect::frame_msg_t *msg) {

    double t = (double)getTickCount();

    //----------------------------------
    // Publish Camera frame
    //----------------------------------
    publish_camera_frame(this, msg->timestamp);

    //----------------------------------
    // Unpack Point cloud
    //----------------------------------
    cv::Mat img, gray;
    cv::Mat_<Vec3f> cloud;
    opencv_utils::unpack_kinect_frame_with_cloud(msg, img, cloud, options.vSCALE);
    // printf("===> UNPACK TIME: %4.2f ms\n", ((double)getTickCount() - t)/getTickFrequency() * 1e3);     

    //--------------------------------------------
    // Gaussian blur, and gray
    //--------------------------------------------
    // cv::GaussianBlur(img, img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor(img, gray, COLOR_BGR2GRAY);
    if (!options.vLIVE_MODE)
        publish_cloud(this, cloud, img);

    // detect April tags (requires a gray scale image)
    vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(gray);

    // Detection
    vs::obj_collection_t objs_msg; 
    objs_msg.id = 10000; 
    objs_msg.name = "POSE_EST"; 
    objs_msg.type = VS_OBJ_COLLECTION_T_AXIS3D; 
    objs_msg.reset = true; 
    objs_msg.objs.resize(detections.size());

    //--------------------------------------------
    // Transform each detection and add to track
    //--------------------------------------------
    // cout << detections.size() << " tags detected:" << endl;

    articulation::pose_list_msg_t pose_list_msg;
    for (int i=0; i<detections.size(); i++) {
        AprilTags::TagDetection& detection = detections[i];

        // Compute Tag Pose
        pose_utils::pose_t tag_pose = compute_tag_pose(this, msg->timestamp, detection, cloud);

        // Unique ID -> idx
        int id; 
        if (id2idx_map.find(detection.id) == id2idx_map.end()) { 
            id = id2idx_map.size();
            id2idx_map[detection.id] = id;
            std::cerr << " remapped (" << detection.id << " to " << id << ")" << std::endl;
        } else { 
            id = id2idx_map[detection.id];
        }


        articulation::pose_msg_t pose_msg;
        pose_msg.utime = msg->timestamp;
        pose_msg.id = detection.id;
        for (int j=0; j<3; j++)
            pose_msg.pos[j] = tag_pose.pos[j];
        // std::cerr << "orientation: " << cv::Mat_<double>(1,4,tag_pose.orientation) << std::endl;
        // FLAG! 
        // for (int j=0; j<4; j++)
        //     pose_msg.orientation[j] = tag_pose.orientation[j];

        // Flipped representation (x,y,z,w)
        pose_msg.orientation[0] = 0, pose_msg.orientation[1] = 0, 
            pose_msg.orientation[2] = 0, pose_msg.orientation[3] = 1;

        //--------------------------------------------
        // Add tag detection to pose_list (publish live)
        //--------------------------------------------
        pose_list_msg.pose.push_back(pose_msg);
        pose_list_msg.num_poses = pose_list_msg.pose.size();

        //--------------------------------------------
        // Add tag detection to tracks (publish batch)
        //--------------------------------------------
        articulation::track_msg_t& track = tracks[id];
        track.id = id;
        track.pose.push_back(pose_msg);
        track.pose_flags.push_back(articulation::track_msg_t::POSE_VISIBLE);
        
        int ch_width = openChannel(track, "width", true);
        int ch_height = openChannel(track, "height", true);

        track.num_poses = track.pose.size();
        track.num_poses_projected = track.pose_projected.size();
        track.num_poses_resampled = track.pose_resampled.size();

        //--------------------------------------------
        // Visualize
        //--------------------------------------------
        // Transform Pose to Sensor Frame
        pose_utils::pose_t tag_pose_tf = transform_to_sensor_frame(this, msg->timestamp, tag_pose);
        double rpy[3]; bot_quat_to_roll_pitch_yaw (tag_pose_tf.orientation, rpy);

        objs_msg.objs[i].id = id; 
        objs_msg.objs[i].x = tag_pose_tf.pos[0], objs_msg.objs[i].y = tag_pose_tf.pos[1], 
            objs_msg.objs[i].z = tag_pose_tf.pos[2];
        objs_msg.objs[i].roll = rpy[0], objs_msg.objs[i].pitch = rpy[1], 
            objs_msg.objs[i].yaw = rpy[2]; 
    }

    // Visualize tag detections 
    objs_msg.nobjs = objs_msg.objs.size(); 
    lcm.publish("OBJ_COLLECTION", &objs_msg);

    //--------------------------------------------
    // Publish tracks (live/batch)
    // live - last pose every frame
    // batch - last 100 poses every k frames
    //--------------------------------------------
    // live case
    if (pose_list_msg.num_poses) { 
      lcm.publish("ARTICULATION_POSE_LIST", &pose_list_msg);
    }
    // batch publish
    publish_batch();

    // show the current image including any detections
    if (options.vDEBUG) {
        // std::cerr << "Detected (" << detections.size() << "): ";
        for (int i=0; i<detections.size(); i++) {
            detections[i].draw(img);
            // std::cerr << detections[i].id << " ";
        }
        // std::cerr << std::endl;
        cv::imshow("AprilTags", img); 
    }

    cv::waitKey(20);

    return;
}

int main(int argc, char** argv)
{
    //----------------------------------
    // Opt args
    //----------------------------------
    ConciseArgs opt(argc, (char**)argv);
    opt.add(options.vLOG_FILENAME, "l", "log-file","Log file name (LIVE: if no log file name)");
    opt.add(options.vCREATE_VIDEO, "v", "create-video","Create video");
    opt.add(options.vCHANNEL, "c", "channel","Kinect Channel");
    opt.add(options.vMAX_FPS, "r", "max-rate","Max FPS");
    opt.add(options.vSCALE, "s", "scale","Scale");    
    opt.add(options.vPLOT, "p", "plot","Plot Viz");
    opt.add(options.vSTART_FRAME, "t", "start-frame","Start frame");
    opt.add(options.vEND_FRAME, "e", "end-frame","End frame");
    opt.add(options.vPUBLISH_EVERY_K_POSES, "k", "k-poses","Publish every K poses");
    opt.parse();

    //----------------------------------
    // args output
    //----------------------------------
    std::cerr << "===========  AprilTags Articulation (Detector) ============" << std::endl;
    std::cerr << "MODES 1: articulation-detector -l log-file -v <create video> -r max-rate\n";
    std::cerr << "=============================================\n";
    opt.usage();
    std::cerr << "===============================================" << std::endl;

    // Handle special args cases
    if (options.vLOG_FILENAME.empty()) { 
        options.vLIVE_MODE = true;
    } else {
        options.vLIVE_MODE = false;
        options.vCREATE_VIDEO = false;

        //----------------------------------
        // Setup Player Options
        //----------------------------------
        poptions.fn = options.vLOG_FILENAME;
        poptions.ch = options.vCHANNEL;
        poptions.fps = options.vMAX_FPS;
        poptions.start_frame = options.vSTART_FRAME;
        poptions.end_frame = options.vEND_FRAME;

        poptions.lcm = &state.lcm;
        poptions.handler = &on_kinect_image_frame;
        poptions.user_data = state.ptr();
    }


    //----------------------------------
    // Subscribe, and start main loop
    //----------------------------------
    if (options.vLIVE_MODE) { 
        state.lcm.subscribe(options.vCHANNEL, &state_t::on_kinect_image_frame, &state );
        std::cerr << " Waiting for kinect frame " << std::endl;
        while (state.lcm.handle() == 0);
    } else { 
        player.init(poptions); 
    }

    // // Not using config file for filter models
    // params.prior_width = 
    //     bot_param_get_double_or_fail(state->param,
    //                                  "articulation_detector.prior_width");
    // params.prior_height = 
    //     bot_param_get_double_or_fail(state->param,
    //                                  "articulation_detector.prior_height");

    return 0;
}


