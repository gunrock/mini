#pragma once

#include "graph.hxx"

namespace gunrock {

enum frontier_type_t {
  edge_frontier = 0,
  node_frontier = 1
};

template<typename type_t>
class frontier_t {
  size_t _size;
  frontier_type_t _type;

  std::shared_ptr<mem_t<type_t> > _data;

public:
  void swap(frontier_t& rhs) {
      std::swap(_size, rhs._size);
      std::swap(_type, rhs._type);
      _data->swap(rhs._data);
  }

  frontier_t() : _size(0), _type(node_frontier), _data(std::make_shared<type_t>())
    { }
  frontier_t& operator=(const frontier_t& rhs) = delete;
  frontier_t(const frontier_t& rhs) = delete;

  frontier_t(context_t &context, size_t capacity, size_t scale, frontier_type_t type = node_frontier) :
      _size(0),
      _type(type)
    {
        int full_size = int(capacity * scale);
        _data.reset(new mem_t<type_t>(full_size, context));
  }

  frontier_t(frontier_t&& rhs) : frontier_t() {
    swap(rhs);
  }
  frontier_t& operator=(frontier_t&& rhs) {
    swap(rhs);
    return *this;
  }

  ~frontier_t() {}

  cudaError_t load(mem_t<type_t> &target) {
      int target_size = target.size();
      int full_size = _data->size();
      if (target_size > full_size) {
        printf("Overflow during frontier loading. Capacity is %d,"
                "size of the data to load is %d.\n",
                full_size,
                target_size);
        exit(0);
      }
      // can fit, load the data into _data
      cudaError_t result = dtod(_data->data(), target.data(), target_size);
      _size = target_size;
      return result;
  }

  cudaError_t load(std::vector<type_t> target) {
    int target_size = target.size();
    int full_size = _data->size();
    if (target_size > full_size) {
        printf("Overflow during frontier loading. Capacity is %d,"
                "size of the data to load is %d.\n",
                full_size,
                target_size);
        exit(0);
    }
    cudaError_t result = htod(_data->data(), target);
    _size = target_size;
    return result;
  }

  //TODO: append array to the end of current data

  size_t size() const { return _size; }
  std::shared_ptr<mem_t<type_t> > data() const {return _data; }
  
};

}
