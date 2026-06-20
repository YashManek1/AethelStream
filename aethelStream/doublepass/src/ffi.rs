//! PyO3 FFI surface for M7 (Python training loop).
//!
//! This module is a placeholder. The full PyO3 bindings land in Sprint 11.
//! Until then, this file exists so the module tree compiles and M7 authors can
//! see the planned surface.
//!
//! S11 will add:
//! ```ignore
//! #[pymodule]
//! fn doublepass_py(_py: Python, m: &PyModule) -> PyResult<()> { ... }
//!
//! #[pyclass]
//! pub struct PyDoublePass { ... }
//!
//! #[pymethods]
//! impl PyDoublePass {
//!     pub fn step(...) -> PyResult<...> { ... }
//!     pub fn set_plan(...) -> PyResult<()> { ... }
//!     pub fn snapshot(...) -> PyResult<...> { ... }
//!     pub fn parity_probe(...) -> PyResult<f64> { ... }
//! }
//! ```
