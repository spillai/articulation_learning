// Articulation Model Learner 
// Author: Sudeep Pillai (mostly stripped from Sturm's work)
// http://www.ros.org/wiki/articulation

// lcm
#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot2_param.h>

// libbot/lcm includes
#include <bot_core/bot_core.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>
#include <bot_param/param_util.h>
#include <bot_lcmgl_client/lcmgl.h>

// opencv-utils includes
#include <perception_opencv_utils/opencv_utils.hpp>

// lcm log player wrapper
#include <lcm-utils/lcm-reader-util.hpp>

// lcm messages
#include <lcmtypes/articulation.hpp> 

// optargs
#include <ConciseArgs>

// visualization msgs
// #include <lcmtypes/visualization.h>
#include <lcmtypes/visualization.hpp>

// #include <lcmtypes/erlcm_interrupt_msg_t.h>

// Articulation model
#include <articulation/factory.h>
#include <articulation/utils.hpp>
#include <boost/foreach.hpp>


using namespace std;
using namespace articulation_models;

// Local 
map<string, double> startingTime;
map<string, vector<double> > measurements;

// State 
struct state_t {
    lcm::LCM lcm;

    BotParam   *b_server;
    BotFrames *b_frames;

    // Factory
    MultiModelFactory factory;

    // Model
    GenericModelVector models_valid;

    state_t() : 
        b_server(NULL),
        b_frames(NULL) {

        // Check lcm
        assert(lcm.good());

        //----------------------------------
        // Bot Param/frames init
        //----------------------------------
        b_server = bot_param_new_from_server(lcm.getUnderlyingLCM(), 1);
        b_frames = bot_frames_get_global (lcm.getUnderlyingLCM(), b_server);
    }
    
    void on_pose_tracks (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                             const articulation::track_msg_t *msg);
    ~state_t() { 
    }

};
state_t state;


struct model_learner_params { 
    double sigma_position;
    double sigma_orientation;
    std::string filter_models;
    model_learner_params() {}
};
model_learner_params model_params;

typedef std::vector<articulation::pose_msg_t> Track;
std::vector<Track> tracks;

void TIC(string name){
    startingTime[name] = timestamp_us();
}

void TOC(string name) {
    measurements[name].push_back( (timestamp_us() - startingTime[name]) );
    if (DEBUG && measurements[name].size())
        std::cerr << "(" << name << ") : " << measurements[name].end()[-1]*1e-3 << " ms " << std::endl;
}

void ADD_DATA(string name,double data) {
    measurements[name].push_back( data );
}
#define SQR(a) ((a)*(a))

void EVAL() {
    map<string, vector<double> >::iterator it;
    for(it = measurements.begin(); it!=measurements.end(); it++) {
        size_t n = it->second.size();
        double sum = 0;
        for(size_t i=0;i<n;i++) {
            sum += it->second[i];
        }
        double mean = sum /n;
        double vsum = 0;
        for(size_t i=0;i<n;i++) {
            vsum += SQR(it->second[i] - mean);
        }
        double var = vsum / n;
        cout << it->first << " " << mean << " "<<sqrt(var)<< " ("<<n<<" obs)"<< endl;
    }
}



// void interrupt() { 
//     erlcm_interrupt_msg_t interrupt;
//     interrupt.source_channel = (char*)"ARTICULATION_OBJECT_TRACKS";
//     erlcm_interrupt_msg_t_publish(state->lcm, "EVENT_INTERRUPT", &interrupt);
// }


void state_t::on_pose_tracks (const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                              const articulation::track_msg_t *msg) {

    std::cerr << "\n-------------- on_pose_tracks -----------" << std::endl;

    std::cerr << "TRACK: " << msg->id << std::endl;

    ArticulationModel model;
    model.track = *msg;

    setParamIfNotDefined(model.params, "sigma_position",
                         model_params.sigma_position, articulation::model_param_msg_t::PRIOR);
    setParamIfNotDefined(model.params, "sigma_orientation",
                         model_params.sigma_orientation, articulation::model_param_msg_t::PRIOR);


    TIC("createModels");
    GenericModelVector models_new = factory.createModels( model );
    TOC("createModels");

    GenericModelVector models_old = models_valid;
  
    models_valid.clear();
    models_old.clear();


    // update old models, then add valid
    for(size_t i=0;i<models_old.size();i++) {
        models_old[i]->setTrack( *msg );
        models_old[i]->projectPoseToConfiguration();
        if( !models_old[i]->fitMinMaxConfigurations() ) continue;
        if( !models_old[i]->evaluateModel() ) continue;

        models_valid.push_back( models_old[i] );
    }

    // fit new models, then add valid
    TIC("per_track");
    for(size_t i=0;i<models_new.size();i++) {
        TIC("fitModel" + models_new[i]->getModelName());
        if( !models_new[i]->fitModel() ) {
            if(DEBUG) cout <<"fitting of "<<models_new[i]->getModelName()<<" failed"<<endl;
            continue;
        }
        TOC("fitModel" + models_new[i]->getModelName());
        TIC("projectPoseToConfiguration" + models_new[i]->getModelName());
        models_new[i]->projectPoseToConfiguration();
        TOC("projectPoseToConfiguration" + models_new[i]->getModelName());
        TIC("fitMinMaxConfigurations" + models_new[i]->getModelName());
        if( !models_new[i]->fitMinMaxConfigurations() ) {
            if(DEBUG) cout <<"fitting of min/max conf of "<<models_new[i]->getModelName()<<" failed"<<endl;
            continue;
        }
        TOC("fitMinMaxConfigurations" + models_new[i]->getModelName());

        TIC("evaluateModel" + models_new[i]->getModelName());
        if( !models_new[i]->evaluateModel() ) {
            if(DEBUG) cout <<"evaluation of "<<models_new[i]->getModelName()<<" failed"<<endl;
            continue;
        }
        TOC("evaluateModel" + models_new[i]->getModelName());

        models_valid.push_back( models_new[i] );

    }
    TOC("per_track");

    map<double,GenericModelPtr> models_sorted;
    for(size_t i=0;i<models_valid.size();i++) {
        if(isnan( models_valid[i]->getBIC() )) {
            if(DEBUG) cout <<"BIC eval of "<<models_new[i]->getModelName()<<" is nan, skipping"<<endl;
            continue;
        }
        models_sorted[models_valid[i]->getBIC()] = models_valid[i];
    }

    if(models_sorted.size()==0) {
        cout << "no valid models found"<<endl;
        return;
    }

    for(map<double,GenericModelPtr>::iterator it=models_sorted.begin();it!=models_sorted.end();it++) {
        cout << it->second->getModelName()<<
            " pos_err=" << it->second->getPositionError()<<
            " rot_err=" << it->second->getOrientationError()<<
            " bic=" << it->second->getBIC()<<
            " k=" << it->second->getParam("complexity") <<
            " n=" << it->second->getTrack().pose.size() <<
            endl;
    }
    //  }
    map<double,GenericModelPtr>::iterator it = models_sorted.begin();
    models_valid.clear();
    models_valid.push_back(it->second);

    //  it->second->projectPoseToConfiguration();
    //  it->second->fitMinMaxConfigurations();
    if(it->second->getModelName()=="rotational") {
        it->second->sampleConfigurationSpace( 0.05 );
    } else {
        it->second->sampleConfigurationSpace( 0.01 );
    }
    return;
}


void get_params() { 

    char* filter_models_str = 
        bot_param_get_str_or_fail(state.b_server, 
                                  "articulation_model_learner.filter_models");
    model_params.filter_models = std::string(filter_models_str);

    model_params.sigma_position = 
        bot_param_get_double_or_fail(state.b_server,
                                     "articulation_model_learner.sigma_position");
    model_params.sigma_orientation = 
        bot_param_get_double_or_fail(state.b_server,
                                     "articulation_model_learner.sigma_orientation");

    cout <<"(param) sigma_position=" << model_params.sigma_position << endl;
    cout <<"(param) sigma_orientation=" << model_params.sigma_orientation << endl;
    cout <<"(param) filter_models=" << model_params.filter_models << endl;

    return;
}

int main(int argc, char** argv)
{
    //----------------------------------
    // Opt args
    //----------------------------------
    ConciseArgs opt(argc, (char**)argv);
    opt.parse();

    //----------------------------------
    // args output
    //----------------------------------
    std::cerr << "===========  Model Learner ============" << std::endl;
    std::cerr << "MODES 1: articulation-model-learner\n";
    std::cerr << "=============================================\n";

    std::cerr << "=> Note: Hit 'space' to proceed to next frame" << std::endl;
    std::cerr << "=> Note: Hit 'p' to proceed to previous frame" << std::endl;
    std::cerr << "=> Note: Hit 'n' to proceed to previous frame" << std::endl;
    std::cerr << "===============================================" << std::endl;

    //----------------------------------
    // Subscribe, and start main loop
    //----------------------------------
    state.lcm.subscribe("ARTICULATION_OBJECT_TRACKS", &state_t::on_pose_tracks, &state);

    // Get Params
    get_params();

    // Set models to be considered
    state.factory.setFilter(model_params.filter_models);
    
    while (state.lcm.handle() == 0);

    return 0;
}


