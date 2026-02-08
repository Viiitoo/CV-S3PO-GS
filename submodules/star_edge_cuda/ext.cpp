#include <torch/extension.h>
#include "star_edge.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_localsh_descriptors", &compute_localsh_descriptors,
          "STAR-Edge LocalSH descriptor computation (CUDA)",
          py::arg("points"),
          py::arg("sh_basis_real"),
          py::arg("sh_basis_imag"),
          py::arg("grid_xyz"),
          py::arg("bw") = 10,
          py::arg("kk") = 26,
          py::arg("num_samples") = 30);

    m.def("knn_search", &knn_search,
          "Morton code based KNN search (CUDA)",
          py::arg("points"),
          py::arg("k") = 26);
}
