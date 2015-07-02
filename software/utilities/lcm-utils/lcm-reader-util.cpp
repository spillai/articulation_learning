#include "lcm-reader-util.hpp"

LCMLogReader::LCMLogReader () {
  internal_init();
}

LCMLogReader::LCMLogReader(const std::string& fn, float scale, bool _sequential_read) {
  internal_init();
  
  //----------------------------------
  // Log player options
  //----------------------------------
  options.lcm =  &lcm;
  options.fn = fn;
  options.scale = scale;
  options.ch = "KINECT_FRAME";
  options.fps = 1000.f;
  options.handler = NULL;
  options.user_data = NULL;

  // Init 
  init(options, false); // sequential_read);

  return;
}

LCMLogReader::~LCMLogReader() { 
    if (log) delete log;
}

void 
LCMLogReader::internal_init() {
    frame_num = 0;
    first_event = 0; 
    log = NULL;
    event = NULL;
}

inline int64_t LCMLogReader::find_closest_utime(int64_t utime) { 
    printf("find_closest_utime: %14.4f\n",utime * 1e-6);
    std::map<int64_t, int64_t>::iterator it = utime_map.lower_bound(utime);
    if (it == utime_map.begin() || it->first == utime) { 
        printf("closest upper: %14.4f\n", it->first * 1e-6);
        return it->second;
    }
    std::map<int64_t, int64_t>::iterator it2 = it;
    it2--;
    if (it2 == utime_map.end() || ((utime - it2->first)  < (it->first - utime))) { 
        printf("closest lower: %14.4f\n", it2->first * 1e-6);
        return it2->second;
    }
    printf("closest upper: %14.4f\n",it->first * 1e-6);
    return it->second;
}

void LCMLogReader::init_index() { 
    std::cerr << "Init indexing" << std::endl;
    // read all events and create map
    // lcm_eventlog_event_t* ev = lcm_eventlog_read_next_event (log);
 
    int count = 0; 
    for (event = log->readNextEvent(); event != NULL; 
         event = log->readNextEvent()) {

      // Save first event utime
      if (first_event == 0) first_event = event->timestamp;
      
      if (options.ch.length() && options.ch == event->channel) { 

        // Decode msg
        kinect::frame_msg_t msg; 
        if(msg.decode(event->data, 0, event->datalen) != event->datalen)
          continue;
        utime_map[int64_t(msg.timestamp)] = int64_t(event->timestamp);
        event_utimes.push_back(int64_t(event->timestamp));
            
        count++;
        // debug
        if (count %100 == 0) { 
          std::cerr << "Indexed " << count << "frames:  " 
                    << msg.timestamp << "->" << event->timestamp << std::endl;
        }
      }
    } 
    std::cerr << "Done indexing" << std::endl;
    assert(utime_map.size() == event_utimes.size());
}

void LCMLogReader::reset() {
  // Delete and reinit (HACK!)
  if (log) delete log;
  log = new lcm::LogFile(options.fn, "r"); // hack  

  // Seek back to first timestamp
  frame_num = 0;
  return;

}

// for c++ calls
void LCMLogReader::init (const LCMLogReaderOptions& _options, bool sequential_read) { 
    // log player options
    options = _options;

    // open log
    log = new lcm::LogFile(options.fn, "r");
    if (!log) {
        fprintf (stderr, "Unable to open source logfile %s\n", options.fn.c_str());
        return;
    }

    std::cerr << "===========  LCM LOG PLAYER ============" << std::endl;
    std::cerr << "=> LOG FILENAME : " << options.fn << std::endl;
    std::cerr << "=> CHANNEL : " << options.ch << std::endl;
    std::cerr << "=> FPS : " << options.fps << std::endl;
    std::cerr << "=> LCM PTR : " << (options.lcm != 0) << std::endl;
    std::cerr << "=> USER_DATA PTR : " << ((options.user_data) ? "AVAILABLE":"NULL") << std::endl;
    std::cerr << "=> START FRAME : " << options.start_frame << std::endl;
    std::cerr << "=> END FRAME : " << options.end_frame << std::endl;
    std::cerr << "=> SCALE : " << options.scale << std::endl;
    std::cerr << "===============================================" << std::endl;

    // Set usleep interval
    usleep_interval = (1.f / options.fps) * 1e4;
    // std::cerr << "usleep: " << usleep_interval << std::endl;

    if (sequential_read)
      getNextFrame(); // start
    else { 
        init_index();

        // reset 
        reset();
    }
    // std::cerr << "First_utime: " << first_event << std::endl;
}

int LCMLogReader::getNumFrames() {
  return utime_map.size();
}

bool LCMLogReader::getNextFrame() { 
    assert(good());
    // assert(options.handler);

    // get next event from log
    event = log->readNextEvent();
    // std::cerr << "getNextFrame: " << event->timestamp << std::endl;
    if (event != NULL) { 
        if (options.ch.length() && options.ch == event->channel) { 
            // Handle start/end frames
            // std::cerr << options.start_frame << " " << options.end_frame << " " << frame_num << std::endl;
            if ((options.end_frame < 0) || 
                (options.start_frame <= frame_num && options.end_frame >= frame_num)) { 
                // Decode msg
                kinect::frame_msg_t msg; 
                if(msg.decode(event->data, 0, event->datalen) != event->datalen)
                    return false;

                // HACK recv_buf_t NULL??
                // std::cerr << "Seeking: " << msg.timestamp << std::endl;
                if (options.handler)
                  options.handler(NULL, event->channel, &msg);

                // For renderers (should'nt subscribe and request next frame at the same time)
                lcm.publish(event->channel, &msg);
                // on_kinect_image_frame(NULL, event->channel, &msg, options.user_data);
            } else { 
                std::cerr << "lcm-reader-util: Processed " << frame_num << " frames: exiting! " << std::endl;
                return true;
            }
            frame_num++;            
        }
    } else { 
        return false;
    }

        
    // Get next frame
    if (options.fps > 0.f) {
      assert(0); // haven't implemented correctly
        usleep(usleep_interval);
        getNextFrame();
    } else  { 
        while(1) { 
          getNextFrame();
        }
    }
    return true;
}

// hack (scale factor)
kinect::frame_msg_t
LCMLogReader::getNextKinectFrame() { 
    assert(good());
    // assert(options.handler);

    kinect::frame_msg_t msg;
    msg.timestamp = 0;
    
    // get next event from log
    while (1) { 
      event = log->readNextEvent();
      // std::cerr << "getNextKinectFrame: " << event->timestamp << std::endl;
      if (event != NULL && options.ch.length() && options.ch == event->channel && 
          msg.decode(event->data, 0, event->datalen) == event->datalen) { 
        // For renderers (should'nt subscribe and request next frame at the same time)
        lcm.publish(event->channel, &msg);

        frame_num++;
        msg.tilt_radians = options.scale; // HACK!!!!!

        return msg;
      }

      if (event == NULL) break;
    }
    return msg;
}

// hack (scale factor)
kinect::frame_msg_t
LCMLogReader::getKinectFrame(double index) { 
    assert(good());
    // assert(options.handler);
    index = std::max(double(0), index-1);
    assert(index >= 0 && index < getNumFrames());
    int64_t event_utime = event_utimes[index];
    // std::cerr << "seek to event_utime: " << event_utime << std::endl;

    kinect::frame_msg_t msg;
    msg.timestamp = 0;
    
    if (log->seekToTimestamp(event_utime) == -1) {
      std::cerr << "Failed to seek to index:" << index << " evutime: " << event_utime << std::endl;
      return msg;
    }

    return getNextKinectFrame();
}

// hack (scale factor)
kinect::frame_msg_t
LCMLogReader::getKinectFrameWithTimestamp(double sensor_utime) { 
    assert(good());
    // assert(options.handler);

    int64_t event_utime = find_closest_utime(sensor_utime);
    std::cerr << "seek to event_utime: " << event_utime << std::endl;

    kinect::frame_msg_t msg;
    msg.timestamp = 0;
    
    if (log->seekToTimestamp(event_utime) == -1) {
      std::cerr << "Failed to seek to " << sensor_utime << " evutime: " << event_utime << std::endl;
      return msg;
    }

    return getNextKinectFrame();
}


