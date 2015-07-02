#include "template.h"
#include "container.h"
#include "opencv_numpy_conversion.hpp"

/*
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder.
 */
namespace fs { namespace opencv {

static void init()
{
    import_array();
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

using namespace cv;
static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

#define OPENCV_3 0
#if OPENCV_3
class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() { stdAllocator = Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
    {
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags) const
    {
        if( data != 0 )
        {
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, int accessFlags) const
    {
        return stdAllocator->allocate(u, accessFlags);
    }

    void deallocate(UMatData* u) const
    {
        if(u)
        {
            PyEnsureGIL gil;
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }
 
    const MatAllocator* stdAllocator;
};
#else
class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        refcount = refcountFromPyObject(o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA((PyArrayObject*) o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};
#endif
  

  
NumpyAllocator g_numpyAllocator;

NDArrayConverter::NDArrayConverter() { init(); }

void NDArrayConverter::init()
{
    import_array();
}

cv::Mat NDArrayConverter::toMat(const PyObject *o)
{
    cv::Mat m;

    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("toMat: Object is not a numpy array");
    }

    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("toMat: Data type = %d is not supported", typenum);
    }

    int ndims = PyArray_NDIM(o);

    if(ndims >= CV_MAX_DIM)
    {
        failmsg("toMat: Dimensionality (=%d) is too high", ndims);
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    bool transposed = false;
    
    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    // std::cerr << " ndims: " << ndims
    //           << " size: " << size
    //           << " type: " << type
    //           << " step: " << step 
    //           << " size: " << size[2] << std::endl;

    // TODO: Possible bug in multi-dimensional matrices
#if 1
    // if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    if( ndims == 3 && size[2] <= 3 && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }
#endif
    
    if( ndims > 2)
    {
        failmsg("toMat: Object has more than 2 dimensions");
    }
    
    m = Mat(ndims, size, type, PyArray_DATA(o), step);

    // std::cerr << " ndims: " << ndims
    //           << " size: " << size
    //           << " type: " << type
    //           << " step: " << step 
    //           << " size: " << size[2] << std::endl;
    // std::cerr << " mat: " << m.rows << " " << m.cols << std::endl;
    // m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    
    if( m.data )
    {
#if OPENCV_3
      m.addref();
      Py_INCREF(o);
#else
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
#endif
    };
    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return m;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m)
{
#if OPENCV_3
  if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->u || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    // p->addref();
    // pyObjectFromRefcount(p->refcount);
    return o; 
#else
    if( !m.data )
      Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
#endif

}

// =======================================================================
// OpenCV <=> Numpy Converters
template <typename T>
struct Mat_to_PyObject {
  static PyObject* convert(const T& v){
    NDArrayConverter cvt;
    PyObject* ret = cvt.toNDArray(cv::Mat(v));
    return ret;
  }
};


// struct Vec3f_to_mat {
//   static PyObject* convert(const cv::Vec3f& v){
//     NDArrayConverter cvt;
//     PyObject* ret = cvt.toNDArray(cv::Mat(v));
//     return ret;
//   }
// };

// struct Point2f_to_mat {
//   static PyObject* convert(const cv::Point2f& v){
//     NDArrayConverter cvt;
//     PyObject* ret = cvt.toNDArray(cv::Mat(v));
//     return ret;
//   }
// };

// struct Point3f_to_mat {
//   static PyObject* convert(const cv::Point3f& v){
//     NDArrayConverter cvt;
//     PyObject* ret = cvt.toNDArray(cv::Mat(v));
//     return ret;
//   }
// };

// // Convert to type T (cv::Mat)
// template <typename T>
// struct Mat_to_pyobject {
//   static PyObject* convert(const T& mat){
//     NDArrayConverter cvt;
//     PyObject* ret = cvt.toNDArray(mat);
//     return ret;
//   }
// };


template <typename T>
struct Mat_python_converter
{
  Mat_python_converter()
  {
    // Register from converter
    boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<T>());

    // Register to converter
    py::to_python_converter<T, Mat_to_PyObject<T> >();
  }

  // Convert from type T to PyObject (numpy array)
  // Assume obj_ptr can be converted in a cv::Mat
  static void* convertible(PyObject* obj_ptr)
  {
    // if (!PyString_Check(obj_ptr)) return 0;
    return obj_ptr;
  }

  // Convert obj_ptr into a cv::Mat
  static void construct(PyObject* obj_ptr,
                        boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    using namespace boost::python;
    typedef converter::rvalue_from_python_storage< T > storage_t;

    storage_t* the_storage = reinterpret_cast<storage_t*>( data );
    void* memory_chunk = the_storage->storage.bytes;

    NDArrayConverter cvt;
    T* newvec = new (memory_chunk) T(cvt.toMat(obj_ptr));
    data->convertible = memory_chunk;

    return;
  }
};
// =======================================================================


void export_converters(void) {

  using namespace fs::opencv;
  // register the to-from-python converter for each of the types
  Mat_python_converter< cv::Mat >();
  
  Mat_python_converter< cv::Mat1b >();
  Mat_python_converter< cv::Mat1s >();
  Mat_python_converter< cv::Mat1w >();
  Mat_python_converter< cv::Mat1i >();
  Mat_python_converter< cv::Mat1f >();
  Mat_python_converter< cv::Mat1d >();

  Mat_python_converter< cv::Mat2b >();
  Mat_python_converter< cv::Mat2f >();
  Mat_python_converter< cv::Mat2d >();
  
  Mat_python_converter< cv::Mat3b >();
  Mat_python_converter< cv::Mat3f >();
  Mat_python_converter< cv::Mat3d >();

  // Expose vector of mats, points etc
  expose_template_type< std::vector<cv::Mat> >();
  expose_template_type< std::vector<cv::Mat_<uchar> > >();
  expose_template_type< std::vector<cv::Mat_<int> > >();
  expose_template_type< std::vector<cv::Mat_<float> > >();
  expose_template_type< std::vector<cv::Mat_<double> > >();

  expose_template_type< std::vector<std::vector< cv::Mat > > >();
  
  expose_template_type< std::vector<cv::Point> >();
  expose_template_type< std::vector<cv::Point2f> >();
  expose_template_type< std::vector<cv::Point3f> >();

  py::to_python_converter<cv::Vec2f, Mat_to_PyObject<cv::Vec2f> >();
  py::to_python_converter<cv::Vec3f, Mat_to_PyObject<cv::Vec3f> >();
  
  expose_template_type< std::vector<cv::KeyPoint> >();

  // Deprecated
  // py::to_python_converter<cv::Point, Point_to_mat>();
  // py::to_python_converter<cv::Point2f, Point2f_to_mat>();
  // py::to_python_converter<cv::Vec3f, Vec3f_to_mat>();
  
  // py::to_python_converter<cv::Point, Mat_to_PyObject<cv::Point> >();
  // py::to_python_converter<cv::Point2f, Mat_to_PyObject<cv::Point2f> >();
  // py::to_python_converter<cv::Point3f, Mat_to_PyObject<cv::Point3f> >();
  // py::to_python_converter<cv::Point_<float>, Mat_to_PyObject<cv::Point_<float> > >();
}

} // namespace opencv 
} // namespace fs
