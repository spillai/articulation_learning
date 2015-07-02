#ifndef THREAD_SAFE_QUEUE_HPP_
#define THREAD_SAFE_QUEUE_HPP_

// boost includes
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <queue>

template <typename T, int N>
class FixedLengthQueue: public std::deque<T> {
 private:
  int max_buffer_size;

 public:

  FixedLengthQueue(): max_buffer_size(N) {}

  void setMaxLength(const int n) {
    max_buffer_size = n;
  }
  
  void push(const T& item) {

    // Push to the back
    this->push_back(item);

    // Max buffer length
    if (this->size() > max_buffer_size) this->pop_front();
  }

  bool full() const
  {
    return this->size() == max_buffer_size;
  }

  void get(T& item, const int index=0)
  {
    assert(index < this->size());
    item=(*this)[index];
  }

  T& get_latest()
  {
    assert(this->size());
    return this->back();
  }

  T& get_oldest()
  {
    assert(this->size());
    return this->front();
  }
  
};

template <typename T>
class ThreadedQueue {
 private:
  std::deque<T> data;
  mutable boost::mutex mutex;
  boost::condition_variable cond;
  int max_buffer_size;

 public:

  ThreadedQueue(const int _max_buffer_size=2): max_buffer_size(2) {
  }

  void push(const T& item) {
    boost::mutex::scoped_lock lock(mutex);

    // Push to the front
    data.push_front(item);

    // Max buffer length
    if (data.size() > max_buffer_size) data.pop_back();

    // Unlock before notifying
    lock.unlock();

    // Notify 
    cond.notify_one();
  }

  bool empty() const
  {
    boost::mutex::scoped_lock lock(mutex);
    return data.empty();
  }


  int size() const
  {
    boost::mutex::scoped_lock lock(mutex);
    return data.size();
  }

  // bool try_pop(T& item)
  // {
  //   boost::mutex::scoped_lock lock(mutex);
  //   if(data.empty())
  //     return false;

  //   item=data.front();
  //   data.pop();
  //   return true;
  // }

  void get(T& item, bool listen=false)
  {
    boost::mutex::scoped_lock lock(mutex);
    while(data.empty())
      cond.wait(lock);

    item=data.back();
    if (!listen)
      data.pop_back();
  }
};
#endif // #ifndef THREAD_SAFE_QUEUE_HPP_

