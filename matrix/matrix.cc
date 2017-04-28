#include <cstdlib>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/algorithms/transform_reduce_binary.hpp>
#include <iterator>
#include <stdexcept>

#include <gtest/gtest.h>


namespace core {

struct Dims {
  int64_t rows;
  int64_t cols;
};

template <typename T> struct matrix_traits;

struct column_major {};
struct row_major {};
template <> struct matrix_traits<column_major> {
  static int64_t &major_dim(Dims &d) { return d.cols; }
  static int64_t &minor_dim(Dims &d) { return d.rows; }
};
template <> struct matrix_traits<row_major> {
  static int64_t &major_dim(Dims &d) { return d.rows; }
  static int64_t &minor_dim(Dims &d) { return d.cols; }
};

class Matrix {

  float *data_;
  static constexpr int64_t ALIGNMENT_ = 64;
  Dims dims_;
  int64_t elements_;

public:
  Matrix();
  Matrix(int64_t rows, int64_t cols)
      : data_(reinterpret_cast<float *>(
            malloc(rows * cols * sizeof(float) + ALIGNMENT_))),
        dims_{rows, cols}, elements_(rows * cols) {}
  Matrix(const Matrix &&src) : Matrix(src.dims_.rows, src.dims_.cols) {
    hpx::parallel::copy(hpx::parallel::execution::par_unseq, src.data_,
                        src.data_ + src.elements_, data_);
  }

  ~Matrix() {
    if (data_)
      free(data_);
  }

  int64_t Rows() const { return dims_.rows; }
  int64_t Cols() const { return dims_.cols; }
  int64_t Element_Count() const { return elements_; }
  float *Data() const { return data_; }

  float &operator()(int64_t r, int64_t c) { return data_[r + c * dims_.rows]; }

  friend core::Matrix operator*(const core::Matrix &, const core::Matrix &);
  friend hpx::future<core::Matrix> operator*(hpx::future<core::Matrix> &&,
                                             hpx::future<core::Matrix> &&);

  //        class Iterator_Base :  public std::iterator<
  //        std::random_access_iterator_tag, float, long, float*, float&>
};

template <typename Traits>
class Iterator_Base : public std::iterator<std::random_access_iterator_tag,
                                           float, long, float *, float &>,
                      private Traits {
  Matrix *m_;
  Dims current_dim_;
  Dims m_dims_;

public:
  explicit Iterator_Base(Matrix *m, Dims dim)
      : m_{m}, current_dim_{dim}, m_dims_{m_->Rows(), m_->Cols()} {}
  explicit Iterator_Base(Matrix *m) : m_{m}, current_dim_{0, 0} {}
  Iterator_Base(const Iterator_Base &src)
      : m_{src.m_}, current_dim_{src.current_dim_}, m_dims_{src.m_dims_} {}
  explicit Iterator_Base() : m_{nullptr} {}
  Iterator_Base &operator++() {
    if (!m_)
      return *this;
    ++Traits::minor_dim(current_dim_);
    if (Traits::minor_dim(current_dim_) == Traits::minor_dim(m_dims_)) {
      Traits::minor_dim(current_dim_) = 0;
      ++Traits::major_dim(current_dim_);
    }
    // current_dim_ contains actual indices.  m_dims_ contains the number of
    // elements in each dimension i.e. for [n x m] matrix, valid current_dim_ is
    // (0,0) -> (n-1, m-1).  Therefore, if after incrementing current_dim_.cols
    // >= m_dims_.cols, it is invalid.
    if (Traits::major_dim(current_dim_) >= Traits::major_dim(m_dims_))
      m_ = nullptr;
    return *this;
  }

  // const Dims Current

  Iterator_Base operator++(int) {
    Iterator_Base new_iter(*this);
    this->operator++();
    return new_iter;
  }

  Iterator_Base &operator+=(int stride) {
    if (!m_)
      return *this;
    int64_t newrow = Traits::minor_dim(current_dim_) + stride;
    Traits::major_dim(current_dim_) += newrow / Traits::minor_dim(m_dims_);
    Traits::minor_dim(current_dim_) =
        abs(newrow % Traits::minor_dim(m_dims_)); // stride *can* be negative

    // The mod operator ensures that the value of rows will always be valid
    // We check the cols to ensure within 0 .. cols

    if (Traits::major_dim(current_dim_) < 0 ||
        Traits::major_dim(current_dim_) >= Traits::major_dim(m_dims_))
      m_ = nullptr;

    return *this;
  }

  Iterator_Base &operator--() {
    if (!m_)
      return *this;
    --Traits::minor_dim(current_dim_);
    if (Traits::minor_dim(current_dim_) < 0) {
      Traits::minor_dim(current_dim_) = Traits::minor_dim(m_dims_);
      --Traits::major_dim(current_dim_);
    }
    if (Traits::major_dim(current_dim_) < 0)
      m_ = nullptr;
    return *this;
  }

  Iterator_Base operator--(int) {
    Iterator_Base new_iter(*this);
    this->operator--();
    return new_iter;
  }

  Iterator_Base &operator-=(int stride) { return this->operator+=(-stride); }

  value_type &operator*() {
    if (m_)
      return m_->operator()(current_dim_.rows, current_dim_.cols);

    throw std::runtime_error ("Error!  Cannot dereference invalid iterator");
  }

  bool operator==(const Iterator_Base &other) {

    // if both are nullptr, doesn't matter what value of current_dims_
    if (!this->m_ && !other.m_)
      return true;
    // one or more points are non-nullptr.  If different, iterators are
    // non-equal
    if (this->m_ != other.m_)
      return false;
    // points to same matrix...
    if (this->current_dim_ != other.current_dim_)
      return false;
    return true;
  }

  bool operator!=(const Iterator_Base &other) { return !(*this == other); }

  const Dims Current_Dim() const { return this->current_dim_; }

  uint64_t address() const {
    return this->m_->operator()(current_dim_.rows, current_dim_.cols);
  }
};

using Col_Iterator = core::Iterator_Base<core::matrix_traits<column_major>>;
using Row_Iterator = core::Iterator_Base<core::matrix_traits<row_major>>;

Col_Iterator begin_cols(Matrix *m, Dims d = {0, 0}) {
  return Col_Iterator(m, d);
}
Col_Iterator end_cols(Matrix *) { return Col_Iterator(); }
Row_Iterator begin_rows(Matrix *m, Dims d = {0, 0}) {
  return Row_Iterator(m, d);
}
Row_Iterator end_rows(Matrix *) { return Row_Iterator(); }

Col_Iterator::difference_type operator-(const Col_Iterator &l,
                                        const Col_Iterator &r) {
  return l.address() - r.address();
}

core::Matrix operator*( core::Matrix &lhs,  core::Matrix &rhs) {

  if (lhs.Cols() != rhs.Rows())
    throw std::runtime_error("Imcompatible Matrix dimensions");

  core::Matrix m{lhs.Rows(), rhs.Cols()};
  Col_Iterator out_iter(&m);

  // Outermost-loop -- columns of lhs and output
  hpx::parallel::for_loop_n_strided(
      hpx::parallel::execution::par, 0, rhs.Cols(), rhs.Rows(),
      [&](auto out_col_idx) {

        hpx::parallel::for_loop_n(
            hpx::parallel::execution::par, 0, lhs.Rows(),
            [&](auto out_row_idx) {

              m(out_row_idx, out_col_idx) = hpx::parallel::transform_reduce(
                  hpx::parallel::execution::par,
                  Row_Iterator(&lhs, {out_row_idx, 0}),
                  Row_Iterator(&lhs, {out_row_idx, lhs.Cols()}),
                  Col_Iterator(&rhs, {0, out_col_idx}), 0.0f,
                  std::plus<float>(),
                  [&](const float &a, const float &b) { return a * b; });
            });

      });
  return m;
}

} // namespace core

int main(int argc, char **argv) { return 0; }
