//! Python FFI bindings via PyO3.

use crate::loader::ShardLoader;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python wrapper for ShardLoader.
#[pyclass]
pub struct PyShardLoader {
    inner: ShardLoader,
}

#[pymethods]
impl PyShardLoader {
    /// Create a new loader for the given model directory.
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let loader = ShardLoader::new(&model_dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyShardLoader { inner: loader })
    }

    /// Load raw safetensors bytes for an entire layer shard.
    fn load_layer<'py>(&mut self, py: Python<'py>, layer_index: u32) -> PyResult<&'py PyBytes> {
        let buf = self
            .inner
            .load_layer(layer_index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &buf.data))
    }

    /// Load FP16 bytes for a single parameter (dequantizes NF4 on the fly).
    fn load_param<'py>(&mut self, py: Python<'py>, param_name: &str) -> PyResult<&'py PyBytes> {
        let buf = self.inner.load_param(param_name).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(msg)
            } else {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(msg)
            }
        })?;
        Ok(PyBytes::new(py, &buf.data))
    }

    /// Get the shape of a parameter as a Python list.
    fn param_shape(&self, param_name: &str) -> PyResult<Vec<usize>> {
        let info = self
            .inner
            .store
            .tensor_info(param_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
        Ok(info.shape.clone())
    }

    /// Get the precision string for a parameter.
    fn param_precision(&self, param_name: &str) -> PyResult<String> {
        let info = self
            .inner
            .store
            .tensor_info(param_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
        Ok(info.precision.clone())
    }
}

/// PyO3 module definition.
#[pymodule]
pub fn shard_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyShardLoader>()?;
    Ok(())
}
