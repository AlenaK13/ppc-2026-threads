#include "kruglova_a_conjugate_gradient_sle/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "kruglova_a_conjugate_gradient_sle/common/include/common.hpp"

namespace kruglova_a_conjugate_gradient_sle {

namespace {

void CalculateDistribution(int n, int mpi_size, int rank, int &local_n, int &offset) {
  int q = n / mpi_size;
  int rem = n % mpi_size;
  if (rank < rem) {
    local_n = q + 1;
    offset = rank * (q + 1);
  } else {
    local_n = q;
    offset = (rank * q) + rem;
  }
}

void ComputeLocalMatVec(int local_n, int n, int offset, const InType &input, const std::vector<double> &p_full,
                        std::vector<double> &local_ap) {
#pragma omp parallel for default(none) shared(local_n, n, offset, input, p_full, local_ap)
  for (int i = 0; i < local_n; ++i) {
    double sum = 0.0;
    size_t row_idx = static_cast<size_t>(offset + i) * static_cast<size_t>(n);
    for (int j = 0; j < n; ++j) {
      sum += input.A[row_idx + static_cast<size_t>(j)] * p_full[j];
    }
    local_ap[i] = sum;
  }
}

}  // namespace

KruglovaAConjGradSleALL::KruglovaAConjGradSleALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KruglovaAConjGradSleALL::ValidationImpl() {
  const auto &in = GetInput();
  if (in.size <= 0) {
    return false;
  }
  if (in.A.size() != static_cast<size_t>(in.size) * static_cast<size_t>(in.size)) {
    return false;
  }
  if (in.b.size() != static_cast<size_t>(in.size)) {
    return false;
  }
  return true;
}

bool KruglovaAConjGradSleALL::PreProcessingImpl() {
  GetOutput().assign(GetInput().size, 0.0);
  return true;
}

bool KruglovaAConjGradSleALL::RunImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  const auto &input = GetInput();
  const int n = input.size;
  auto &x_global = GetOutput();

  int local_n = 0;
  int offset = 0;
  CalculateDistribution(n, mpi_size, rank, local_n, offset);

  std::vector<double> local_r(local_n);
  std::vector<double> local_p(local_n);
  std::vector<double> local_ap(local_n);
  std::vector<double> local_x(local_n, 0.0);
  std::vector<double> p_full(n);

  std::vector<int> send_counts(mpi_size);
  std::vector<int> displs(mpi_size);
  int q = n / mpi_size;
  int rem = n % mpi_size;
  for (int i = 0; i < mpi_size; ++i) {
    send_counts[i] = (i < rem) ? (q + 1) : q;
    displs[i] = (i < rem) ? (i * (q + 1)) : ((i * q) + rem);
  }

#pragma omp parallel for default(none) shared(local_n, offset, local_r, local_p, input)
  for (int i = 0; i < local_n; ++i) {
    local_r[i] = input.b[static_cast<size_t>(offset + i)];
    local_p[i] = local_r[i];
  }

  double local_rsold = 0.0;
#pragma omp parallel for default(none) shared(local_n, local_r) reduction(+ : local_rsold)
  for (int i = 0; i < local_n; ++i) {
    local_rsold += local_r[i] * local_r[i];
  }
  double rsold = 0.0;
  MPI_Allreduce(&local_rsold, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double tolerance = 1e-8;
  for (int iter = 0; iter < n; ++iter) {
    MPI_Allgatherv(local_p.data(), local_n, MPI_DOUBLE, p_full.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    ComputeLocalMatVec(local_n, n, offset, input, p_full, local_ap);

    double local_p_ap = 0.0;
#pragma omp parallel for default(none) shared(local_n, local_p, local_ap) reduction(+ : local_p_ap)
    for (int i = 0; i < local_n; ++i) {
      local_p_ap += local_p[i] * local_ap[i];
    }
    double p_ap = 0.0;
    MPI_Allreduce(&local_p_ap, &p_ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::abs(p_ap) < 1e-16) {
      break;
    }

    const double alpha = rsold / p_ap;
    double local_rsnew = 0.0;
#pragma omp parallel for default(none) shared(local_n, local_x, local_p, local_r, local_ap, alpha) \
    reduction(+ : local_rsnew)
    for (int i = 0; i < local_n; ++i) {
      local_x[i] += alpha * local_p[i];
      local_r[i] -= alpha * local_ap[i];
      local_rsnew += local_r[i] * local_r[i];
    }

    double rsnew = 0.0;
    MPI_Allreduce(&local_rsnew, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (std::sqrt(rsnew) < tolerance) {
      break;
    }

    const double beta = rsnew / rsold;
#pragma omp parallel for default(none) shared(local_n, local_p, local_r, beta)
    for (int i = 0; i < local_n; ++i) {
      local_p[i] = local_r[i] + (beta * local_p[i]);
    }
    rsold = rsnew;
  }

  MPI_Allgatherv(local_x.data(), local_n, MPI_DOUBLE, x_global.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);
  return true;
}

bool KruglovaAConjGradSleALL::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_conjugate_gradient_sle
