// mini gunrock, copyright (c) 2016, Yangzihao Wang
#pragma once
#include "meta.hxx"

BEGIN_GUNROCK_NAMESPACE

namespace detail {

template<typename it_t,
  typename type_t = typename std::iterator_traits<it_t>::value_type,
  bool use_ldg =
    std::is_pointer<it_t>::value &&
    std::is_arithmetic<type_t>::value
>
struct ldg_load_t {
  GUNROCK_HOST_DEVICE static type_t load(it_t it) {
    return *it;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350

template<typename it_t, typename type_t>
struct ldg_load_t<it_t, type_t, true> {
  GUNROCK_HOST_DEVICE static type_t load(it_t it) {
    return __ldg(it);
  }
};

#endif

} // namespace detail

template<typename it_t>
GUNROCK_HOST_DEVICE typename std::iterator_traits<it_t>::value_type
ldg(it_t it) {
  return detail::ldg_load_t<it_t>::load(it);
}

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators.

template<typename type_t>
struct less_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a < b;
  }
};
template<typename type_t>
struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a <= b;
  }
};
template<typename type_t>
struct greater_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a > b;
  }
};
template<typename type_t>
struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a >= b;
  }
};
template<typename type_t>
struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a == b;
  }
};
template<typename type_t>
struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
  GUNROCK_HOST_DEVICE bool operator()(type_t a, type_t b) const {
    return a != b;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators.

template<typename type_t>
struct plus_t : public std::binary_function<type_t, type_t, type_t> {
	GUNROCK_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a + b;
  }
};

template<typename type_t>
struct minus_t : public std::binary_function<type_t, type_t, type_t> {
	GUNROCK_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a - b;
  }
};

template<typename type_t>
struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
  GUNROCK_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return a * b;
  }
};

template<typename type_t>
struct maximum_t  : public std::binary_function<type_t, type_t, type_t> {
  GUNROCK_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return max(a, b);
  }
};

template<typename type_t>
struct minimum_t  : public std::binary_function<type_t, type_t, type_t> {
  GUNROCK_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
    return min(a, b);
  }
};

////////////////////////////////////////////////////////////////////////////////
// iterator_t and const_iterator_t are base classes for customized iterators.

template<typename outer_t, typename int_t, typename value_type>
struct iterator_t : public std::iterator_traits<const value_type*> {

  iterator_t() = default;
  GUNROCK_HOST_DEVICE iterator_t(int_t i) : index(i) { }

  GUNROCK_HOST_DEVICE outer_t operator+(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next += diff;
    return next;
  }
  GUNROCK_HOST_DEVICE outer_t operator-(int_t diff) const {
    outer_t next = *static_cast<const outer_t*>(this);
    next -= diff;
    return next;
  }
  GUNROCK_HOST_DEVICE outer_t& operator+=(int_t diff) {
    index += diff;
    return *static_cast<outer_t*>(this);
  }
  GUNROCK_HOST_DEVICE outer_t& operator-=(int_t diff) {
    index -= diff;
    return *static_cast<outer_t*>(this);
  }

  int_t index;
};

template<typename outer_t, typename int_t, typename value_type>
struct const_iterator_t : public iterator_t<outer_t, int_t, value_type> {
  typedef iterator_t<outer_t, int_t, value_type> base_t;

  const_iterator_t() = default;
  GUNROCK_HOST_DEVICE const_iterator_t(int_t i) : base_t(i) { }

  // operator[] and operator* are tagged as DEVICE-ONLY.  This is to ensure
  // compatibility with lambda capture in CUDA 7.5, which does not support
  // marking a lambda as __host__ __device__.
  // We hope to relax this when a future CUDA fixes this problem.
  GUNROCK_DEVICE value_type operator[](int_t diff) const {
    return static_cast<const outer_t&>(*this)(base_t::index + diff);
  }
  GUNROCK_DEVICE value_type operator*() const {
    return (*this)[0];
  }
};

////////////////////////////////////////////////////////////////////////////////
// discard_iterator_t is a store iterator that discards its input.

template<typename value_type>
struct discard_iterator_t :
  iterator_t<discard_iterator_t<value_type>, int, value_type> {

  struct assign_t {
    GUNROCK_HOST_DEVICE value_type operator=(value_type v) {
      return value_type();
    }
  };

  GUNROCK_HOST_DEVICE assign_t operator[](int index) const {
    return assign_t();
  }
  GUNROCK_HOST_DEVICE assign_t operator*() const { return assign_t(); }
};

////////////////////////////////////////////////////////////////////////////////
// counting_iterator_t returns index.

template<typename type_t, typename int_t = int>
struct counting_iterator_t :
  const_iterator_t<counting_iterator_t<type_t>, int_t, type_t> {

  counting_iterator_t() = default;
  GUNROCK_HOST_DEVICE counting_iterator_t(type_t i) :
    const_iterator_t<counting_iterator_t, int_t, type_t>(i) { }

  GUNROCK_HOST_DEVICE type_t operator()(int_t index) const {
    return (type_t)index;
  }
};

////////////////////////////////////////////////////////////////////////////////
// strided_iterator_t returns offset + index * stride.

template<typename type_t, typename int_t = int>
struct strided_iterator_t :
  const_iterator_t<strided_iterator_t<type_t>, int_t, int> {

  strided_iterator_t() = default;
  GUNROCK_HOST_DEVICE strided_iterator_t(type_t offset_, type_t stride_) :
    const_iterator_t<strided_iterator_t, int_t, type_t>(0),
    offset(offset_), stride(stride_) { }

  GUNROCK_HOST_DEVICE type_t operator()(int_t index) const {
    return offset + index * stride;
  }

  type_t offset, stride;
};

////////////////////////////////////////////////////////////////////////////////
// constant_iterator_t returns the value it was initialized with.

template<typename type_t>
struct constant_iterator_t :
  const_iterator_t<constant_iterator_t<type_t>, int, type_t> {

  type_t value;

  GUNROCK_HOST_DEVICE constant_iterator_t(type_t value_) : value(value_) { }

  GUNROCK_HOST_DEVICE type_t operator()(int index) const {
    return value;
  }
};

END_GUNROCK_NAMESPACE
