#ifndef THREAD_SAFE_GRABBER_HPP_
#define THREAD_SAFE_GRABBER_HPP_

// Thread safe queue
#include "thread_safe_queue.hpp"

// boost includes
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/function.hpp>

// std includes
#include <thread>
#include <queue>

template <typename T>
class ThreadSafeGrabber {
 private:
  ThreadedQueue<T> queue;

  std::thread lcm_th;
  std::thread process_th;

 public:
  lcm::LCM lcm;  
  std::string channel;

  typedef void (run_cb_ptr) (const T& msg);
  typedef void (run_cb) (const T& msg);
  boost::function<run_cb> run;
  
 public:
  ThreadSafeGrabber(const std::string _channel = "KINECT_FRAME"): channel(_channel) {
    // Start process and lcm handler threads
    lcm_th = std::thread(&ThreadSafeGrabber::lcm_thread_handler, this);
    process_th = std::thread(&ThreadSafeGrabber::process_thread_handler, this);
  }

  ~ThreadSafeGrabber() {
    process_th.join();
    lcm_th.join();
  }

  void on_data(const lcm::ReceiveBuffer* rbuf, const std::string& chan,
               const T *msg) {
    queue.push(*msg);
  }

  void setCallback(void* f, void* obj) {
    // run = boost::bind(f, obj, _1);
  }
  
  void subscribe(const std::string channel) {
    // Subscribe
    lcm.subscribe(channel, &ThreadSafeGrabber::on_data, this );
  }
  
  void lcm_thread_handler() {
    while (lcm.handle() == 0);
  }

  void process_thread_handler() {
    while (1) { 
      T msg;
      queue.get(msg);
      run(msg);
    }
  }
};

#endif // #ifndef THREAD_SAFE_GRABBER_HPP_

